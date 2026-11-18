"""Filesystem layout and atomic-write helpers.

Durability / crash-safety rules implemented here:
  * data is always written to a temp file first, fsync'ed, then atomically
    renamed into place (os.replace), followed by an fsync of the directory;
  * a partially written file therefore never appears at its final path;
  * `replace` is the single choke point for the final rename so tests can
    simulate disk failures by monkeypatching it.

Cold archival lives in a separate volume:
  * each cold volume has a private `.staging/` directory; a cold copy is
    written there, fsync'ed and digest-checked, then atomically renamed to
    `<volume>/<task_id>`, so a cold path is either absent or a verified,
    complete copy;
  * cold and hot are different filesystems, so a cold copy is always a copy
    plus rename, never a cross-device `os.replace` of the hot file;
  * downloads keep an open file descriptor (`open_download`): unlinking the
    hot file afterwards does not affect a transfer already in progress.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

READ_BUF = 1024 * 1024
VOLUME_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class VerificationError(Exception):
    """Size or sha256 of a file did not match the recorded package."""


def replace(src: str, dst: str) -> None:
    """Atomic rename. Monkeypatched by tests to simulate disk failure."""
    os.replace(src, dst)


def fsync_dir(path) -> None:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def unlink_quiet(path) -> None:
    try:
        os.unlink(str(path))
    except OSError:
        pass


def write_temp(tmp_dir: Path, data: bytes) -> Path:
    """Write *data* to a fresh temp file in *tmp_dir* (fsync'ed)."""
    fd, tmp = tempfile.mkstemp(dir=str(tmp_dir), prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        unlink_quiet(tmp)
        raise
    return Path(tmp)


def file_size_and_digest(path) -> tuple[int, str]:
    """Return (size, sha256 hex) of *path*, read in bounded chunks."""
    digest = hashlib.sha256()
    total = 0
    with open(str(path), "rb") as f:
        while True:
            buf = f.read(READ_BUF)
            if not buf:
                break
            digest.update(buf)
            total += len(buf)
    return total, digest.hexdigest()


def verify_file(path, expected_size: int, expected_sha256: str) -> None:
    """Raise VerificationError unless *path* matches size and digest."""
    try:
        size, digest = file_size_and_digest(path)
    except OSError as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    if size != expected_size or digest != expected_sha256:
        raise VerificationError(
            f"verification failed: size {size} (expected {expected_size}),"
            f" sha256 {digest} (expected {expected_sha256})"
        )


def copy_verified(
    src,
    staging,
    expected_size: int,
    expected_sha256: str,
    delay_ms: float = 0.0,
) -> None:
    """Copy *src* to *staging*, fsync it and verify size+sha256.

    The staging file is removed on any failure, so a caller that sees this
    raise never has a partial staging file to reason about. `delay_ms` is a
    test aid used to widen the crash window in the live tests.
    """
    digest = hashlib.sha256()
    total = 0
    try:
        with open(str(src), "rb") as fin, open(str(staging), "wb") as fout:
            while True:
                buf = fin.read(READ_BUF)
                if not buf:
                    break
                fout.write(buf)
                digest.update(buf)
                total += len(buf)
                if delay_ms:
                    time.sleep(delay_ms / 1000.0)
            fout.flush()
            os.fsync(fout.fileno())
    except BaseException:
        unlink_quiet(staging)
        raise

    if total != expected_size or digest.hexdigest() != expected_sha256:
        unlink_quiet(staging)
        raise VerificationError(
            f"source verification failed: size {total}"
            f" (expected {expected_size}), sha256 {digest.hexdigest()}"
            f" (expected {expected_sha256})"
        )


class ColdVolume:
    """One mounted cold volume (a directory on a dedicated filesystem)."""

    def __init__(self, name: str, root) -> None:
        self.name = name
        self.root = Path(root)

    @property
    def staging_dir(self) -> Path:
        return self.root / ".staging"

    def task_path(self, task_id: str) -> Path:
        return self.root / task_id

    def staging_path(self, task_id: str) -> Path:
        return self.staging_dir / f"{task_id}.{uuid.uuid4().hex}.tmp"

    def is_available(self) -> bool:
        try:
            return self.root.is_dir() and os.access(self.root, os.W_OK | os.X_OK)
        except OSError:
            return False

    def free_bytes(self) -> int:
        try:
            return shutil.disk_usage(str(self.root)).free
        except OSError:
            return -1

    def ensure_staging_dir(self) -> None:
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def cleanup_stale_staging(self, max_age_sec: float = 3600.0) -> None:
        """Remove orphaned staging files older than *max_age_sec*.

        Age-based, so a freshly started process never deletes a staging file
        that a concurrently running process is still writing.
        """
        if not self.staging_dir.is_dir():
            return
        now = time.time()
        for p in self.staging_dir.iterdir():
            try:
                if p.is_file() and now - p.stat().st_mtime > max_age_sec:
                    p.unlink()
            except OSError:
                pass


class ColdVolumes:
    """Set of configured cold volumes, parsed from `COLD_VOLUMES`.

    Format: `name=/path[,name=/path ...]`. A bare path gets a generated
    name. Volume names are stored in the DB and shown on the page.
    """

    def __init__(self, volumes: list[ColdVolume] | None = None) -> None:
        self.volumes = list(volumes or [])
        names = [v.name for v in self.volumes]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate cold volume names: {names}")

    @classmethod
    def parse(cls, spec: str | None) -> "ColdVolumes":
        spec = (spec or "").strip()
        volumes: list[ColdVolume] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                name, path = part.split("=", 1)
                name, path = name.strip(), path.strip()
            else:
                path = part
                name = f"cold-{len(volumes) + 1}"
            if not VOLUME_NAME_RE.match(name):
                raise ValueError(f"invalid cold volume name: {name!r}")
            if not path:
                raise ValueError(f"cold volume {name!r} has no path")
            volumes.append(ColdVolume(name, Path(path).resolve()))
        return cls(volumes)

    @classmethod
    def from_env(cls) -> "ColdVolumes":
        return cls.parse(os.environ.get("COLD_VOLUMES", ""))

    def get(self, name: str | None) -> ColdVolume | None:
        if not name:
            return None
        for v in self.volumes:
            if v.name == name:
                return v
        return None

    def available(self) -> list[ColdVolume]:
        return [v for v in self.volumes if v.is_available()]

    def choose(self, preferred: str | None = None) -> ColdVolume | None:
        """Pick a writable volume.

        An explicitly requested volume must exist and be writable; otherwise
        the caller gets None. Without a preference the volume with the most
        free space wins (ties broken by configured order).
        """
        if preferred:
            vol = self.get(preferred)
            if vol is None or not vol.is_available():
                return None
            return vol
        candidates = self.available()
        if not candidates:
            return None
        best = candidates[0]
        for v in candidates[1:]:
            if v.free_bytes() > best.free_bytes():
                best = v
        return best

    def ensure_staging_dirs(self) -> None:
        for v in self.available():
            try:
                v.ensure_staging_dir()
            except OSError:
                pass

    def cleanup_stale_staging(self, max_age_sec: float = 3600.0) -> None:
        for v in self.available():
            v.cleanup_stale_staging(max_age_sec)


class Storage:
    """Path layout under the data root."""

    def __init__(self, root, colds: ColdVolumes | None = None) -> None:
        self.root = Path(root)
        self.db_path = self.root / "uploader.db"
        self.chunks_dir = self.root / "chunks"
        self.published_dir = self.root / "published"
        self.tmp_dir = self.root / "tmp"
        self.locks_dir = self.root / "locks"
        self.colds = colds or ColdVolumes()

    def ensure_dirs(self) -> None:
        for d in (self.chunks_dir, self.published_dir, self.tmp_dir, self.locks_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.colds.ensure_staging_dirs()

    def chunk_path(self, task_id: str, index: int) -> Path:
        return self.chunks_dir / task_id / f"{index:08d}.part"

    def published_path(self, task_id: str) -> Path:
        return self.published_dir / task_id

    def lock_path(self, task_id: str) -> Path:
        return self.locks_dir / f"{task_id}.lock"

    def cleanup_stale_tmp(self, max_age_sec: float = 3600.0) -> None:
        """Remove orphaned temp files older than *max_age_sec*.

        Age-based so a freshly started process never deletes a temp file
        that a concurrently running process is about to rename.
        """
        now = time.time()
        for pattern in (".tmp-*", ".pub-*"):
            for p in self.tmp_dir.glob(pattern):
                try:
                    if now - p.stat().st_mtime > max_age_sec:
                        p.unlink()
                except OSError:
                    pass
        self.colds.cleanup_stale_staging(max_age_sec)
