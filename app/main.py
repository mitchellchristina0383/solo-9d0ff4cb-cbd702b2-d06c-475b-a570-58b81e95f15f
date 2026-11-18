"""Resumable chunked upload service for incident diagnostic packages.

Flow:
  1. POST /api/tasks            create a task with filename, size, sha256
  2. PUT  /api/tasks/{id}/chunks/{i}   upload 1 MiB chunks (idempotent)
  3. POST /api/tasks/{id}/finalize     verify chunks+size+sha256, then publish
  4. GET  /api/tasks/{id}/download     download the published package
  5. POST /api/tasks/{id}/archive      move a published package to a cold volume

Safety properties:
  * a chunk recorded in the DB and present on disk is immutable: re-uploading
    identical content is a no-op, different content is rejected with 409 and
    the original chunk is kept;
  * finalization is serialized per task across processes with an flock, the
    published file appears only via an atomic rename after full
    size+sha256 verification, so a crash or a disk error can never publish a
    partial file and two processes can never produce two different products;
  * all state lives in SQLite (WAL) and on disk, so restarts resume cleanly
    and already published packages stay downloadable;
  * cold archival is serialized per task with the same flock and follows a
    strict order: verified copy on the cold volume -> durable routing switch
    -> hot copy deletion. Download routing only ever flips hot -> cold after
    a verified cold copy exists, and downloads hold an open descriptor, so a
    transfer that started before (or during) the migration always reads a
    complete file. A failed or interrupted archive leaves the hot copy and
    hot routing untouched and can simply be retried.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import re
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from . import storage as storage_mod
from .db import connect, init_db
from .storage import ColdVolumes, Storage

CHUNK_SIZE = 1024 * 1024  # 1 MiB
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
STATIC_DIR = Path(__file__).resolve().parent / "static"
READ_BUF = 1024 * 1024
# Pause between streaming chunks in the download path. 0 in production; the
# live tests start the server with GSB_DOWNLOAD_DELAY_MS to force the archive
# to run while a download is mid-flight.
DOWNLOAD_DELAY_MS = float(os.environ.get("GSB_DOWNLOAD_DELAY_MS", "0") or 0)
# Test aid: widens the window between copying a cold file and recording it.
ARCHIVE_DELAY_MS = float(os.environ.get("GSB_ARCHIVE_DELAY_MS", "0") or 0)


class CorruptChunk(Exception):
    def __init__(self, index: int):
        super().__init__(f"chunk {index} corrupted on disk")
        self.index = index


class ArchiveError(Exception):
    """A cold archive attempt failed; the hot copy is still authoritative."""

    def __init__(self, detail: str, status_code: int = 500):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class CreateTask(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(ge=0)
    sha256: str


class ArchiveRequest(BaseModel):
    volume: str | None = Field(default=None, max_length=64)


class PackageResponse(Response):
    """Stream an already-open package file descriptor.

    The descriptor is opened before any headers are sent, so archival may
    unlink/replace the path afterwards without affecting this transfer. The
    descriptor is closed in a `finally` block, which runs deterministically on
    completion, error and client disconnect.
    """

    chunk_size = READ_BUF

    def __init__(self, fd: int, size: int, filename: str, volume: str):
        self.fd = fd
        encoded = quote(filename)
        disposition = (
            f"attachment; filename*=utf-8''{encoded}"
            if encoded != filename
            else f'attachment; filename="{filename}"'
        )
        super().__init__(
            media_type="application/octet-stream",
            headers={
                "content-length": str(size),
                "content-disposition": disposition,
                "x-package-volume": volume,
            },
        )

    async def __call__(self, scope, receive, send) -> None:
        try:
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                }
            )
            while True:
                buf = await anyio.to_thread.run_sync(os.read, self.fd, self.chunk_size)
                if not buf:
                    break
                await send(
                    {"type": "http.response.body", "body": buf, "more_body": True}
                )
                if DOWNLOAD_DELAY_MS:
                    await anyio.sleep(DOWNLOAD_DELAY_MS / 1000.0)
            await send(
                {"type": "http.response.body", "body": b"", "more_body": False}
            )
        finally:
            try:
                os.close(self.fd)
            except OSError:
                pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_filename(name: str) -> str:
    name = name.replace("\\", "/").split("/")[-1].strip().strip("\x00")
    return (name or "package.bin")[:200]


def _expected_chunk_size(task, index: int) -> int:
    if index < task["total_chunks"] - 1:
        return task["chunk_size"]
    return task["size"] - task["chunk_size"] * (task["total_chunks"] - 1)


def create_app(data_dir=None, cold_volumes=None) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("DATA_DIR", "data")).resolve()
    if cold_volumes is None:
        cold_volumes = ColdVolumes.from_env()
    elif not isinstance(cold_volumes, ColdVolumes):
        # Accept a plain mapping/list of (name, path) for convenience.
        cold_volumes = ColdVolumes(
            [
                storage_mod.ColdVolume(name, path)
                for name, path in (
                    cold_volumes.items()
                    if hasattr(cold_volumes, "items")
                    else cold_volumes
                )
            ]
        )
    storage = Storage(data_dir, cold_volumes)
    storage.ensure_dirs()
    init_db(storage.db_path)
    storage.cleanup_stale_tmp()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # A previous process may have been killed in the middle of an archive.
        # Its staging file is garbage (the copy was never recorded or routed),
        # so drop it and clear the 'copying' marker. The per-task flock tells
        # a dead attempt from one that is still running in another process:
        # only rows whose lock can be taken are touched.
        conn = connect(storage.db_path)
        try:
            rows = conn.execute(
                "SELECT id, archive_staging FROM tasks WHERE archive_state = 'copying'"
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            lock_path = storage.lock_path(row["id"])
            try:
                lock_file = open(lock_path, "w")
            except OSError:
                continue
            try:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue  # a live process owns this archive attempt
                if row["archive_staging"]:
                    storage_mod.unlink_quiet(row["archive_staging"])
                conn = connect(storage.db_path)
                try:
                    conn.execute(
                        "UPDATE tasks SET archive_state = NULL,"
                        " archive_staging = NULL WHERE id = ?"
                        " AND archive_state = 'copying'",
                        (row["id"],),
                    )
                finally:
                    conn.close()
            finally:
                lock_file.close()
        yield

    app = FastAPI(title="Incident Package Uploader", lifespan=lifespan)

    # ------------------------------------------------------------- helpers
    def get_task_or_404(conn, task_id: str):
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "task not found")
        return row

    def _archive_fields(task) -> dict:
        """Archive-related view of a task row (never claims more than the DB)."""
        is_published = task["state"] == "published"
        cold = task["storage"] == "cold"
        return {
            "storage": task["storage"],
            "volume": task["archive_target"] if cold else None,
            "archive_state": task["archive_state"] if is_published else None,
            "archive_target": task["archive_target"],
            "archive_error": task["archive_error"],
            "archived_at": task["archived_at"],
            "hot_deleted": bool(task["archive_deleted_hot"]),
        }

    def status_payload(conn, task) -> dict:
        rows = conn.execute(
            "SELECT chunk_index FROM chunks WHERE task_id = ? ORDER BY chunk_index",
            (task["id"],),
        ).fetchall()
        received = [r["chunk_index"] for r in rows]
        received_set = set(received)
        missing = [i for i in range(task["total_chunks"]) if i not in received_set]
        payload = {
            "task_id": task["id"],
            "filename": task["filename"],
            "size": task["size"],
            "sha256": task["sha256"],
            "chunk_size": task["chunk_size"],
            "total_chunks": task["total_chunks"],
            "received_count": len(received),
            "received_chunks": received,
            "missing_chunks": missing,
            "state": task["state"],
            "created_at": task["created_at"],
            "published_at": task["published_at"],
            **_archive_fields(task),
        }
        if task["state"] == "published":
            payload["download_url"] = f"/api/tasks/{task['id']}/download"
        return payload

    # ------------------------------------------------------------- API
    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.post("/api/tasks", status_code=201)
    def create_task(body: CreateTask):
        sha = body.sha256.lower()
        if not SHA256_RE.match(sha):
            raise HTTPException(400, "sha256 must be 64 hex characters")
        total_chunks = (body.size + CHUNK_SIZE - 1) // CHUNK_SIZE
        task_id = uuid.uuid4().hex
        conn = connect(storage.db_path)
        try:
            conn.execute(
                "INSERT INTO tasks (id, filename, size, sha256, chunk_size,"
                " total_chunks, state, created_at) VALUES (?,?,?,?,?,?, 'uploading', ?)",
                (
                    task_id,
                    _sanitize_filename(body.filename),
                    body.size,
                    sha,
                    CHUNK_SIZE,
                    total_chunks,
                    _now_iso(),
                ),
            )
            task = get_task_or_404(conn, task_id)
            return JSONResponse(status_code=201, content=status_payload(conn, task))
        finally:
            conn.close()

    @app.get("/api/tasks")
    def list_tasks():
        conn = connect(storage.db_path)
        try:
            rows = conn.execute(
                "SELECT t.*, (SELECT COUNT(*) FROM chunks c WHERE c.task_id = t.id)"
                " AS received FROM tasks t ORDER BY created_at DESC"
            ).fetchall()
            return {
                "tasks": [
                    {
                        "task_id": r["id"],
                        "filename": r["filename"],
                        "size": r["size"],
                        "sha256": r["sha256"],
                        "total_chunks": r["total_chunks"],
                        "received_count": r["received"],
                        "state": r["state"],
                        "created_at": r["created_at"],
                        "published_at": r["published_at"],
                        **_archive_fields(r),
                    }
                    for r in rows
                ]
            }
        finally:
            conn.close()

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str):
        conn = connect(storage.db_path)
        try:
            return status_payload(conn, get_task_or_404(conn, task_id))
        finally:
            conn.close()

    @app.put("/api/tasks/{task_id}/chunks/{index}")
    async def put_chunk(task_id: str, index: int, request: Request):
        data = await request.body()
        conn = connect(storage.db_path)
        try:
            task = get_task_or_404(conn, task_id)
            if task["state"] != "uploading":
                raise HTTPException(
                    409, f"task is '{task['state']}'; not accepting chunks"
                )
            if not 0 <= index < task["total_chunks"]:
                raise HTTPException(400, "chunk index out of range")
            expected = _expected_chunk_size(task, index)
            if len(data) != expected:
                raise HTTPException(
                    400, f"chunk size mismatch: expected {expected}, got {len(data)}"
                )

            digest = hashlib.sha256(data).hexdigest()
            final_path = storage.chunk_path(task_id, index)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = storage_mod.write_temp(storage.tmp_dir, data)

            # Record first, then rename. The PRIMARY KEY serializes concurrent
            # uploads of the same chunk across processes: exactly one INSERT
            # wins, and only the winner's content reaches the final path.
            try:
                conn.execute(
                    "INSERT INTO chunks (task_id, chunk_index, sha256, size)"
                    " VALUES (?,?,?,?)",
                    (task_id, index, digest, len(data)),
                )
                inserted = True
            except sqlite3.IntegrityError:
                inserted = False

            if not inserted:
                row = conn.execute(
                    "SELECT sha256 FROM chunks WHERE task_id = ? AND chunk_index = ?",
                    (task_id, index),
                ).fetchone()
                if row["sha256"] != digest:
                    storage_mod.unlink_quiet(tmp)
                    raise HTTPException(
                        409,
                        "chunk already exists with different content;"
                        " original chunk kept",
                    )
                if not final_path.exists():
                    # Heal: row exists but a previous crash lost the file.
                    try:
                        storage_mod.replace(str(tmp), str(final_path))
                        storage_mod.fsync_dir(final_path.parent)
                    except OSError as exc:
                        storage_mod.unlink_quiet(tmp)
                        raise HTTPException(500, f"disk write failed: {exc}")
                else:
                    storage_mod.unlink_quiet(tmp)
                return {"status": "ok", "index": index, "deduplicated": True}

            try:
                storage_mod.replace(str(tmp), str(final_path))
                storage_mod.fsync_dir(final_path.parent)
            except OSError as exc:
                conn.execute(
                    "DELETE FROM chunks WHERE task_id = ? AND chunk_index = ?",
                    (task_id, index),
                )
                storage_mod.unlink_quiet(tmp)
                raise HTTPException(500, f"disk write failed: {exc}")
            return {"status": "ok", "index": index, "deduplicated": False}
        finally:
            conn.close()

    @app.post("/api/tasks/{task_id}/finalize")
    def finalize(task_id: str):
        # Serialize finalization of this task across all service processes.
        lock_file = open(storage.lock_path(task_id), "w")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            conn = connect(storage.db_path)
            try:
                task = get_task_or_404(conn, task_id)
                if task["state"] == "published":
                    return {
                        "state": "published",
                        "download_url": f"/api/tasks/{task_id}/download",
                        "detail": "already published",
                    }
                if task["state"] == "failed":
                    raise HTTPException(
                        409,
                        "task previously failed digest verification;"
                        " please create a new task",
                    )

                rows = conn.execute(
                    "SELECT chunk_index, sha256 FROM chunks WHERE task_id = ?",
                    (task_id,),
                ).fetchall()
                hashes = {r["chunk_index"]: r["sha256"] for r in rows}

                missing = []
                for i in range(task["total_chunks"]):
                    if i not in hashes:
                        missing.append(i)
                    elif not storage.chunk_path(task_id, i).exists():
                        # DB row without file (crash between insert and
                        # rename): drop the row so the chunk can be re-sent.
                        conn.execute(
                            "DELETE FROM chunks WHERE task_id = ? AND chunk_index = ?",
                            (task_id, i),
                        )
                        missing.append(i)
                if missing:
                    return JSONResponse(
                        status_code=409,
                        content={
                            "state": "uploading",
                            "missing_chunks": missing,
                            "detail": "chunks missing; upload them and finalize again",
                        },
                    )

                # Assemble into a temp file; rename to the published path
                # only AFTER size+sha256 verification passes.
                total = 0
                digest = hashlib.sha256()
                tmp = storage.tmp_dir / f".pub-{uuid.uuid4().hex}"
                try:
                    with open(tmp, "wb") as out:
                        for i in range(task["total_chunks"]):
                            chunk_hash = hashlib.sha256()
                            with open(storage.chunk_path(task_id, i), "rb") as f:
                                while True:
                                    buf = f.read(READ_BUF)
                                    if not buf:
                                        break
                                    out.write(buf)
                                    digest.update(buf)
                                    chunk_hash.update(buf)
                                    total += len(buf)
                            if chunk_hash.hexdigest() != hashes[i]:
                                raise CorruptChunk(i)
                        out.flush()
                        os.fsync(out.fileno())
                except CorruptChunk as cc:
                    storage_mod.unlink_quiet(tmp)
                    conn.execute(
                        "DELETE FROM chunks WHERE task_id = ? AND chunk_index = ?",
                        (task_id, cc.index),
                    )
                    return JSONResponse(
                        status_code=409,
                        content={
                            "state": "uploading",
                            "missing_chunks": [cc.index],
                            "detail": "chunk corrupted on disk; please re-upload it",
                        },
                    )
                except OSError as exc:
                    storage_mod.unlink_quiet(tmp)
                    raise HTTPException(500, f"disk write failed: {exc}")

                actual = digest.hexdigest()
                if total != task["size"] or actual != task["sha256"]:
                    storage_mod.unlink_quiet(tmp)
                    conn.execute(
                        "UPDATE tasks SET state = 'failed' WHERE id = ?", (task_id,)
                    )
                    return JSONResponse(
                        status_code=409,
                        content={
                            "state": "failed",
                            "expected_sha256": task["sha256"],
                            "actual_sha256": actual,
                            "expected_size": task["size"],
                            "actual_size": total,
                            "detail": "size or sha256 mismatch; task rejected,"
                            " please create a new task",
                        },
                    )

                # Verification passed: atomically publish, then record state.
                published_path = storage.published_path(task_id)
                published_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    storage_mod.replace(str(tmp), str(published_path))
                    storage_mod.fsync_dir(published_path.parent)
                except OSError as exc:
                    storage_mod.unlink_quiet(tmp)
                    raise HTTPException(500, f"disk write failed: {exc}")

                conn.execute(
                    "UPDATE tasks SET state = 'published', published_at = ?"
                    " WHERE id = ?",
                    (_now_iso(), task_id),
                )
                return {
                    "state": "published",
                    "download_url": f"/api/tasks/{task_id}/download",
                }
            finally:
                conn.close()
        finally:
            lock_file.close()

    def _snapshot_download(task_id: str):
        """Pick the authoritative file for a download and open it.

        The open descriptor is what makes archival safe for in-flight
        downloads: deleting the hot file does not affect a reader that
        already opened it, and a task's routing only flips to cold after the
        verified cold copy exists. If routing flips between the DB read and
        the open, retry against the newly routed location.
        """
        for _ in range(4):
            conn = connect(storage.db_path)
            try:
                task = get_task_or_404(conn, task_id)
            finally:
                conn.close()
            if task["state"] != "published":
                raise HTTPException(
                    409, f"task is '{task['state']}'; not downloadable"
                )
            path, problem = _resolve_archive_location(task)
            if path is None:
                raise HTTPException(500, problem)
            try:
                fd = os.open(str(path), os.O_RDONLY)
            except FileNotFoundError:
                continue  # routing flipped just now; re-read and retry
            except OSError as exc:
                raise HTTPException(500, f"cannot open package: {exc}")
            # Confirm the descriptor still belongs to the routed location.
            conn = connect(storage.db_path)
            try:
                now_task = get_task_or_404(conn, task_id)
            finally:
                conn.close()
            if now_task["storage"] == task["storage"] and (
                now_task["archive_target"] == task["archive_target"]
            ):
                return task, fd
            os.close(fd)
        raise HTTPException(500, "package location kept changing; retry download")

    @app.get("/api/tasks/{task_id}/download")
    def download(task_id: str):
        task, fd = _snapshot_download(task_id)
        return PackageResponse(
            fd,
            task["size"],
            task["filename"],
            task["archive_target"] if task["storage"] == "cold" else "hot",
        )
    def _resolve_archive_location(task) -> tuple[Path | None, str | None]:
        """Where a published package currently is, per the DB routing column.

        Returns (path, problem). A non-None *problem* means the authoritative
        copy is missing on disk; callers must not guess another location.
        """
        if task["storage"] == "cold":
            vol = storage.colds.get(task["archive_target"])
            if vol is None:
                return None, (
                    f"cold volume '{task['archive_target']}' is not configured"
                )
            path = vol.task_path(task["id"])
            if not path.is_file():
                return None, (
                    f"package is routed to cold volume '{vol.name}' but the"
                    " file is missing there"
                )
            return path, None
        path = storage.published_path(task["id"])
        if not path.is_file():
            return None, "published file missing on disk"
        return path, None

    def _delete_hot(conn, task_id: str) -> bool:
        """Delete the hot copy once routing is cold and the cold copy verified.

        Idempotent and safe to call on every retry: if the file is already
        gone it just records that fact. A failure here leaves the hot file in
        place (harmless: routing is already cold) and returns False.
        """
        hot = storage.published_path(task_id)
        try:
            if hot.exists():
                os.unlink(str(hot))
                storage_mod.fsync_dir(hot.parent)
            conn.execute(
                "UPDATE tasks SET archive_deleted_hot = 1 WHERE id = ?", (task_id,)
            )
            return True
        except (OSError, sqlite3.Error):
            return False

    def _archive_error_if_untouched(conn, task) -> bool:
        """Whether a failure can be recorded without contradicting the DB.

        Re-read the row: another process may have archived the task after we
        read it. Only report a failure when the task is still published, still
        routed hot and no archive is in progress.
        """
        if task["state"] != "published":
            return False
        fresh = conn.execute(
            "SELECT state, storage, archive_state, archive_deleted_hot"
            " FROM tasks WHERE id = ?",
            (task["id"],),
        ).fetchone()
        return (
            fresh is not None
            and fresh["state"] == "published"
            and fresh["storage"] == "hot"
            and not fresh["archive_deleted_hot"]
            and fresh["archive_state"] in (None, "copying")
        )

    def _do_archive(task, conn, volume: str | None) -> dict:
        """Copy a published package to a cold volume, then switch routing.

        Caller holds the per-task flock. Ordering is deliberate:

          1. verify the hot source (size + sha256);
          2. copy to the cold volume's staging dir, fsync, verify the copy;
          3. look for an existing cold copy: if one already exists and
             verifies, keep it (the loser of a cross-process race adopts the
             winner's copy instead of writing a second one);
          4. atomically rename staging -> <volume>/<task_id>;
          5. durably record archive_state='archived' (cold copy is now the
             recorded second copy);
          6. durably switch `storage` to 'cold' (downloads now route cold);
          7. only now delete the hot file.

        At every crash point either the hot copy still exists and is routed,
        or the cold copy is verified and routed. A retry re-runs the whole
        sequence and is idempotent.
        """
        task_id = task["id"]
        if task["state"] != "published":
            raise ArchiveError(
                f"task is '{task['state']}'; only published packages can be archived",
                409,
            )
        if task["storage"] == "cold":
            hot_deleted = _delete_hot(conn, task_id)
            return {
                "state": "archived",
                "storage": "cold",
                "volume": task["archive_target"],
                "hot_deleted": hot_deleted,
                "detail": (
                    "already archived; hot copy cleaned up" if hot_deleted
                    else "already archived"
                ),
            }
        if task["archive_state"] == "archived":
            # Crash window: the verified cold copy was recorded but the
            # routing switch did not happen. Downloads still use the hot copy,
            # which was never deleted. Check the recorded cold copy again and
            # finish the switch; if the cold copy is not usable, fall back to
            # a clean retry (the hot copy is untouched).
            vol = storage.colds.get(task["archive_target"])
            cold_path = vol.task_path(task_id) if vol else None
            cold_ok = False
            if cold_path is not None and cold_path.is_file():
                try:
                    storage_mod.verify_file(cold_path, task["size"], task["sha256"])
                    cold_ok = True
                except storage_mod.VerificationError:
                    cold_ok = False
            if not cold_ok:
                conn.execute(
                    "UPDATE tasks SET archive_state = NULL, archive_error = ?"
                    " WHERE id = ?",
                    ("previous cold copy is missing or invalid; retrying", task_id),
                )
            else:
                conn.execute(
                    "UPDATE tasks SET storage = 'cold' WHERE id = ?", (task_id,)
                )
                hot_deleted = _delete_hot(conn, task_id)
                return {
                    "state": "archived",
                    "storage": "cold",
                    "volume": vol.name,
                    "hot_deleted": hot_deleted,
                    "detail": "recovered interrupted archive; routing completed",
                }

        if volume:
            # An explicitly requested volume must be configured and writable;
            # an auto-chosen volume is picked from the currently writable set.
            vol = storage.colds.get(volume)
            if vol is not None and not vol.is_available():
                vol = None
        else:
            vol = storage.colds.choose()
        if vol is None:
            raise ArchiveError(
                f"cold volume '{volume}' is unavailable" if volume
                else "no cold volume is available",
                503,
            )
        try:
            vol.ensure_staging_dir()
        except OSError as exc:
            raise ArchiveError(f"cold volume '{vol.name}' is not writable: {exc}", 503)

        hot = storage.published_path(task_id)
        if not hot.is_file():
            raise ArchiveError("published file missing on disk", 500)

        cold_final = vol.task_path(task_id)
        staging = cold_final.parent / ".staging" / f"{task_id}.{uuid.uuid4().hex}.tmp"

        # Another process may have finished between our lock and this read
        # (e.g. it completed and we were waiting on the flock). Routing is
        # already cold, so any hot file left behind is redundant.
        fresh = get_task_or_404(conn, task_id)
        if fresh["storage"] == "cold":
            return {
                "state": "archived",
                "storage": "cold",
                "volume": fresh["archive_target"],
                "hot_deleted": _delete_hot(conn, task_id),
                "detail": "already archived",
            }

        conn.execute(
            "UPDATE tasks SET archive_state = 'copying', archive_target = ?,"
            " archive_source = ?, archive_staging = ?, archive_error = NULL"
            " WHERE id = ?",
            (vol.name, str(hot), str(staging), task_id),
        )
        try:
            # 1+2: verified copy into the cold staging area.
            try:
                storage_mod.copy_verified(
                    hot, staging, task["size"], task["sha256"],
                    delay_ms=ARCHIVE_DELAY_MS,
                )
            except (OSError, storage_mod.VerificationError) as exc:
                raise ArchiveError(f"cold copy failed: {exc}", 500)

            # 3: adopt a copy that a previous attempt landed but never
            # recorded (crash between rename and the DB update), or that a
            # racing process landed. A cold file that does not verify can only
            # be an unrecorded leftover: routing is still hot, so the verified
            # hot source remains authoritative and the leftover is replaced.
            existing_ok = False
            if cold_final.is_file():
                try:
                    storage_mod.verify_file(cold_final, task["size"], task["sha256"])
                    existing_ok = True
                except storage_mod.VerificationError:
                    existing_ok = False
            if existing_ok:
                storage_mod.unlink_quiet(staging)
            else:
                # 4: atomically publish the verified cold copy.
                try:
                    storage_mod.replace(str(staging), str(cold_final))
                    storage_mod.fsync_dir(cold_final.parent)
                except OSError as exc:
                    storage_mod.unlink_quiet(staging)
                    raise ArchiveError(f"cold volume write failed: {exc}", 500)

            # 5: durably record the verified cold copy.
            conn.execute(
                "UPDATE tasks SET archive_state = 'archived', archive_target = ?,"
                " archive_staging = NULL, archived_at = COALESCE(archived_at, ?)"
                " WHERE id = ?",
                (vol.name, _now_iso(), task_id),
            )

            # 6: durably switch download routing to the cold copy.
            conn.execute(
                "UPDATE tasks SET storage = 'cold', archive_target = ?"
                " WHERE id = ?",
                (vol.name, task_id),
            )
        except BaseException as exc:
            # Never leave a half-copied artifact. The hot copy is never
            # touched before step 7, so it stays authoritative and routed.
            storage_mod.unlink_quiet(staging)
            detail = getattr(exc, "detail", None)
            try:
                if detail is not None:
                    conn.execute(
                        "UPDATE tasks SET archive_state = NULL,"
                        " archive_staging = NULL, archive_error = ? WHERE id = ?",
                        (detail[:500], task_id),
                    )
                else:
                    conn.execute(
                        "UPDATE tasks SET archive_state = NULL,"
                        " archive_staging = NULL WHERE id = ?",
                        (task_id,),
                    )
            except sqlite3.Error:
                pass
            if isinstance(exc, OSError):
                raise ArchiveError(f"cold copy failed: {exc}", 500) from exc
            raise

        # 7: routing is cold and durable; the hot copy is now redundant.
        # The cold copy is verified and routed, so a failure here only leaves
        # a harmless extra hot file that a retry deletes.
        hot_deleted = _delete_hot(conn, task_id)
        return {
            "state": "archived",
            "storage": "cold",
            "volume": vol.name,
            "hot_deleted": hot_deleted,
            "detail": None if hot_deleted else "hot copy cleanup failed; retry archiving",
        }

    @app.post("/api/tasks/{task_id}/archive")
    def archive(task_id: str, body: ArchiveRequest | None = None):
        volume = body.volume if body else None
        lock_file = open(storage.lock_path(task_id), "w")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            conn = connect(storage.db_path)
            try:
                task = get_task_or_404(conn, task_id)
                try:
                    return _do_archive(task, conn, volume)
                except ArchiveError as exc:
                    if _archive_error_if_untouched(conn, task):
                        # Pre-flight failures (no volume, unwritable volume) or
                        # our own failed attempt deserve a user-visible
                        # diagnostic; the hot copy was never touched.
                        try:
                            conn.execute(
                                "UPDATE tasks SET archive_error = ? WHERE id = ?"
                                " AND storage = 'hot'"
                                " AND (archive_state IS NULL"
                                "      OR archive_state != 'archived')",
                                (exc.detail[:500], task_id),
                            )
                        except sqlite3.Error:
                            pass
                    raise HTTPException(exc.status_code, exc.detail)
                except OSError as exc:
                    raise HTTPException(500, f"archive failed: {exc}")
            finally:
                conn.close()
        finally:
            lock_file.close()

    @app.get("/api/storage")
    def storage_info():
        return {
            "volumes": [
                {
                    "name": v.name,
                    "path": str(v.root),
                    "available": v.is_available(),
                    "free_bytes": v.free_bytes(),
                }
                for v in storage.colds.volumes
            ]
        }

    @app.get("/", response_class=HTMLResponse)
    def index_page():
        return HTMLResponse(
            (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        )

    return app


app = create_app()
