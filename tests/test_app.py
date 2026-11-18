"""Functional tests (TestClient): flow, resume, idempotency, conflict,
finalize verification, disk-failure injection, and restart recovery."""
import hashlib
import random
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import storage as storage_mod  # noqa: E402
from app.main import CHUNK_SIZE, create_app  # noqa: E402
from app.storage import ColdVolumes  # noqa: E402


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make_client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path / "data"))


def make_cold_client(tmp_path: Path, name="cold-a") -> tuple[TestClient, Path, Path]:
    """App with one cold volume; returns (client, hot data dir, cold dir)."""
    data_dir = tmp_path / "data"
    cold_dir = tmp_path / "cold" / name
    cold_dir.mkdir(parents=True)
    colds = ColdVolumes([storage_mod.ColdVolume(name, cold_dir)])
    return TestClient(create_app(data_dir, colds)), data_dir, cold_dir


def make_task(client: TestClient, data: bytes, filename="pkg.bin", sha=None) -> str:
    r = client.post(
        "/api/tasks",
        json={"filename": filename, "size": len(data), "sha256": sha or sha256(data)},
    )
    assert r.status_code == 201, r.text
    return r.json()["task_id"]


def chunks_of(data: bytes):
    return [data[i : i + CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]


def put(client, tid, idx, payload):
    return client.put(f"/api/tasks/{tid}/chunks/{idx}", content=payload)


def upload_all(client, tid, data):
    for i, c in enumerate(chunks_of(data)):
        r = put(client, tid, i, c)
        assert r.status_code == 200, (i, r.text)


DATA = random.Random(42).randbytes(2 * CHUNK_SIZE + 123)  # 3 chunks


def test_full_flow(tmp_path):
    c = make_client(tmp_path)
    tid = make_task(c, DATA, filename="diag.tar.gz")
    upload_all(c, tid, DATA)

    st = c.get(f"/api/tasks/{tid}").json()
    assert st["state"] == "uploading"
    assert st["received_count"] == 3
    assert st["missing_chunks"] == []

    r = c.post(f"/api/tasks/{tid}/finalize")
    assert r.status_code == 200 and r.json()["state"] == "published"

    dl = c.get(f"/api/tasks/{tid}/download")
    assert dl.status_code == 200
    assert dl.content == DATA
    assert "diag.tar.gz" in dl.headers["content-disposition"]


def test_resume_and_status(tmp_path):
    c = make_client(tmp_path)
    data = random.Random(1).randbytes(3 * CHUNK_SIZE + 7)  # 4 chunks
    tid = make_task(c, data)
    put(c, tid, 0, chunks_of(data)[0])
    put(c, tid, 2, chunks_of(data)[2])

    st = c.get(f"/api/tasks/{tid}").json()
    assert st["received_chunks"] == [0, 2]
    assert st["missing_chunks"] == [1, 3]

    put(c, tid, 1, chunks_of(data)[1])
    put(c, tid, 3, chunks_of(data)[3])
    assert c.post(f"/api/tasks/{tid}/finalize").json()["state"] == "published"
    assert c.get(f"/api/tasks/{tid}/download").content == data


def test_duplicate_same_content_is_idempotent(tmp_path):
    c = make_client(tmp_path)
    tid = make_task(c, DATA)
    first = chunks_of(DATA)[0]
    r1 = put(c, tid, 0, first)
    r2 = put(c, tid, 0, first)
    assert r1.status_code == 200 and r1.json()["deduplicated"] is False
    assert r2.status_code == 200 and r2.json()["deduplicated"] is True
    st = c.get(f"/api/tasks/{tid}").json()
    assert st["received_chunks"] == [0]


def test_conflicting_content_rejected_original_kept(tmp_path):
    c = make_client(tmp_path)
    tid = make_task(c, DATA)
    original = chunks_of(DATA)[0]
    assert put(c, tid, 0, original).status_code == 200

    evil = bytearray(original)
    evil[0] ^= 0xFF
    r = put(c, tid, 0, bytes(evil))
    assert r.status_code == 409

    upload_all(c, tid, DATA)  # chunk 0 dedup-hits the original
    assert c.post(f"/api/tasks/{tid}/finalize").json()["state"] == "published"
    assert c.get(f"/api/tasks/{tid}/download").content == DATA


def test_wrong_chunk_size_and_index(tmp_path):
    c = make_client(tmp_path)
    tid = make_task(c, DATA)
    assert put(c, tid, 0, b"too short").status_code == 400
    assert put(c, tid, 99, chunks_of(DATA)[0]).status_code == 400
    assert put(c, "0" * 32, 0, chunks_of(DATA)[0]).status_code == 404


def test_finalize_with_missing_chunks_then_complete(tmp_path):
    c = make_client(tmp_path)
    tid = make_task(c, DATA)
    put(c, tid, 0, chunks_of(DATA)[0])

    r = c.post(f"/api/tasks/{tid}/finalize")
    assert r.status_code == 409
    body = r.json()
    assert body["state"] == "uploading"
    assert body["missing_chunks"] == [1, 2]
    # 缺块可补传
    put(c, tid, 1, chunks_of(DATA)[1])
    put(c, tid, 2, chunks_of(DATA)[2])
    assert c.post(f"/api/tasks/{tid}/finalize").json()["state"] == "published"


def test_digest_mismatch_rejects_publish(tmp_path):
    c = make_client(tmp_path)
    tid = make_task(c, DATA, sha=sha256(b"not the real content"))
    upload_all(c, tid, DATA)

    r = c.post(f"/api/tasks/{tid}/finalize")
    assert r.status_code == 409
    assert r.json()["state"] == "failed"

    st = c.get(f"/api/tasks/{tid}").json()
    assert st["state"] == "failed"
    # 拒绝发布：published 目录不得留有成品
    assert not (tmp_path / "data" / "published" / tid).exists()
    assert list((tmp_path / "data" / "published").iterdir()) == []
    # 失败后：不可下载、不可再传、不可再提交
    assert c.get(f"/api/tasks/{tid}/download").status_code == 409
    assert put(c, tid, 0, chunks_of(DATA)[0]).status_code == 409
    assert c.post(f"/api/tasks/{tid}/finalize").status_code == 409


def test_published_task_is_immutable_and_finalize_idempotent(tmp_path):
    c = make_client(tmp_path)
    tid = make_task(c, DATA)
    upload_all(c, tid, DATA)
    assert c.post(f"/api/tasks/{tid}/finalize").json()["state"] == "published"

    assert put(c, tid, 0, chunks_of(DATA)[0]).status_code == 409
    r = c.post(f"/api/tasks/{tid}/finalize")
    assert r.status_code == 200 and r.json()["state"] == "published"
    assert c.get(f"/api/tasks/{tid}/download").content == DATA


def test_disk_failure_on_chunk_upload_not_recorded(tmp_path, monkeypatch):
    c = make_client(tmp_path)

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(storage_mod, "replace", boom)
    r = put(c, tid := make_task(c, DATA), 0, chunks_of(DATA)[0])
    assert r.status_code == 500
    st = c.get(f"/api/tasks/{tid}").json()
    assert st["received_chunks"] == []  # 未记入

    monkeypatch.undo()
    assert put(c, tid, 0, chunks_of(DATA)[0]).status_code == 200


def test_disk_failure_on_finalize_keeps_uploading(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    tid = make_task(c, DATA)
    upload_all(c, tid, DATA)

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(storage_mod, "replace", boom)
    r = c.post(f"/api/tasks/{tid}/finalize")
    assert r.status_code == 500
    assert c.get(f"/api/tasks/{tid}").json()["state"] == "uploading"
    assert not (tmp_path / "data" / "published" / tid).exists()  # 无残缺成品

    monkeypatch.undo()
    assert c.post(f"/api/tasks/{tid}/finalize").json()["state"] == "published"
    assert c.get(f"/api/tasks/{tid}/download").content == DATA


def test_restart_resumes_and_published_survives(tmp_path):
    data = random.Random(9).randbytes(2 * CHUNK_SIZE + 5)
    c1 = make_client(tmp_path)
    tid = make_task(c1, data)
    put(c1, tid, 0, chunks_of(data)[0])

    # 进程重启：同一数据目录新建 app
    c2 = make_client(tmp_path)
    st = c2.get(f"/api/tasks/{tid}").json()
    assert st["received_chunks"] == [0]
    put(c2, tid, 1, chunks_of(data)[1])
    put(c2, tid, 2, chunks_of(data)[2])
    assert c2.post(f"/api/tasks/{tid}/finalize").json()["state"] == "published"

    c3 = make_client(tmp_path)
    assert c3.get(f"/api/tasks/{tid}/download").content == data


def test_empty_file(tmp_path):
    c = make_client(tmp_path)
    tid = make_task(c, b"")
    st = c.get(f"/api/tasks/{tid}").json()
    assert st["total_chunks"] == 0
    assert c.post(f"/api/tasks/{tid}/finalize").json()["state"] == "published"
    assert c.get(f"/api/tasks/{tid}/download").content == b""


def test_validation(tmp_path):
    c = make_client(tmp_path)
    assert c.post("/api/tasks", json={
        "filename": "x", "size": 1, "sha256": "zz"}).status_code == 400
    assert c.post("/api/tasks", json={
        "filename": "x", "size": -1, "sha256": sha256(b"")}).status_code == 422
    assert c.get(f"/api/tasks/{'0'*32}").status_code == 404
    assert c.post(f"/api/tasks/{'0'*32}/finalize").status_code == 404
    assert c.get(f"/api/tasks/{'0'*32}/download").status_code == 404


def test_index_page_served(tmp_path):
    c = make_client(tmp_path)
    r = c.get("/")
    assert r.status_code == 200 and "断点续传" in r.text


# --------------------------------------------------------------- cold archive


def publish(c, data, **kw):
    tid = make_task(c, data, **kw)
    upload_all(c, tid, data)
    assert c.post(f"/api/tasks/{tid}/finalize").json()["state"] == "published"
    return tid


def test_archive_moves_package_and_keeps_download_url(tmp_path):
    c, data_dir, cold_dir = make_cold_client(tmp_path)
    tid = publish(c, DATA, filename="diag.tar.gz")

    # 归档前列出热卷
    st = c.get(f"/api/tasks/{tid}").json()
    assert st["storage"] == "hot" and st["volume"] is None

    r = c.post(f"/api/tasks/{tid}/archive", json={"volume": "cold-a"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["storage"] == "cold" and body["volume"] == "cold-a"
    assert body["hot_deleted"] is True

    # 冷卷上是校验过的副本；热卷成品已删除；暂存区为空
    assert (cold_dir / tid).read_bytes() == DATA
    assert not (data_dir / "published" / tid).exists()
    assert list((data_dir / "published").iterdir()) == []
    assert list((cold_dir / ".staging").iterdir()) == []

    # 页面显示所在卷；下载地址与内容都不变
    st = c.get(f"/api/tasks/{tid}").json()
    assert st["storage"] == "cold"
    assert st["volume"] == "cold-a"
    assert st["archive_state"] == "archived"
    assert st["hot_deleted"] is True
    assert st["archived_at"]
    listed = {t["task_id"]: t for t in c.get("/api/tasks").json()["tasks"]}
    assert listed[tid]["volume"] == "cold-a"
    dl = c.get(f"/api/tasks/{tid}/download")
    assert dl.status_code == 200 and dl.content == DATA
    assert dl.headers["x-package-volume"] == "cold-a"

    # 归档幂等
    again = c.post(f"/api/tasks/{tid}/archive", json={"volume": "cold-a"})
    assert again.status_code == 200
    assert again.json()["storage"] == "cold"
    assert (cold_dir / tid).read_bytes() == DATA
    assert len([p for p in cold_dir.iterdir() if not p.name.startswith(".")]) == 1


def test_archive_without_cold_volume_is_unavailable(tmp_path):
    c = make_client(tmp_path)  # no COLD_VOLUMES
    tid = publish(c, DATA)
    r = c.post(f"/api/tasks/{tid}/archive", json={})
    assert r.status_code == 503
    assert c.get(f"/api/tasks/{tid}/download").content == DATA
    assert c.get(f"/api/tasks/{tid}").json()["storage"] == "hot"


def test_archive_rejects_unpublished_and_unknown(tmp_path):
    c, data_dir, _ = make_cold_client(tmp_path)
    tid = make_task(c, DATA)
    upload_all(c, tid, DATA)
    assert c.post(f"/api/tasks/{tid}/archive", json={}).status_code == 409
    assert c.post(f"/api/tasks/{'0'*32}/archive", json={}).status_code == 404
    published = publish(c, DATA)
    assert c.post(f"/api/tasks/{published}/archive",
                  json={"volume": "../evil"}).status_code == 503
    assert c.get(f"/api/tasks/{published}").json()["storage"] == "hot"
    assert c.get(f"/api/tasks/{tid}").json()["storage"] == "hot"


def test_archive_disk_failure_keeps_hot_copy_and_retry_succeeds(
    tmp_path, monkeypatch
):
    c, data_dir, cold_dir = make_cold_client(tmp_path)
    tid = publish(c, DATA)

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(storage_mod, "replace", boom)
    r = c.post(f"/api/tasks/{tid}/archive", json={})
    assert r.status_code == 500
    # 仍走热卷、热卷文件完好、冷卷无成品、暂存残留已清理
    st = c.get(f"/api/tasks/{tid}").json()
    assert st["storage"] == "hot" and st["archive_state"] is None
    assert st["archive_error"] and "disk full" in st["archive_error"]
    assert (data_dir / "published" / tid).read_bytes() == DATA
    assert not (cold_dir / tid).exists()
    assert list((cold_dir / ".staging").iterdir()) == []
    assert c.get(f"/api/tasks/{tid}/download").content == DATA

    monkeypatch.undo()
    r = c.post(f"/api/tasks/{tid}/archive", json={})
    assert r.status_code == 200 and r.json()["storage"] == "cold"
    assert c.get(f"/api/tasks/{tid}").json()["archive_error"] is None
    assert c.get(f"/api/tasks/{tid}/download").content == DATA


def test_archive_copy_failure_removes_staging(tmp_path, monkeypatch):
    c, data_dir, cold_dir = make_cold_client(tmp_path)
    tid = publish(c, DATA)

    def failing_copy(*args, **kwargs):
        # Simulate a disk failure part-way through the cold copy.
        staging = args[1]
        Path(staging).write_bytes(b"partial")
        raise OSError("no space left on device")

    monkeypatch.setattr(storage_mod, "copy_verified", failing_copy)
    r = c.post(f"/api/tasks/{tid}/archive", json={})
    assert r.status_code == 500
    assert not (cold_dir / tid).exists()
    assert list((cold_dir / ".staging").iterdir()) == []
    assert (data_dir / "published" / tid).read_bytes() == DATA
    assert c.get(f"/api/tasks/{tid}/download").content == DATA


def test_archive_verification_failure_does_not_publish(tmp_path, monkeypatch):
    c, data_dir, cold_dir = make_cold_client(tmp_path)
    tid = publish(c, DATA)

    # Corrupt the recorded digest's expectation by corrupting the source right
    # before the copy: the copy must fail verification and never be routed.
    def corrupt_copy(src, staging, expected_size, expected_sha256, delay_ms=0.0):
        Path(staging).write_bytes(b"\0" * expected_size)
        from app.storage import verify_file, VerificationError

        try:
            verify_file(staging, expected_size, expected_sha256)
        except VerificationError:
            storage_mod.unlink_quiet(staging)
            raise
        raise AssertionError("should not verify")

    monkeypatch.setattr(storage_mod, "copy_verified", corrupt_copy)
    r = c.post(f"/api/tasks/{tid}/archive", json={})
    assert r.status_code == 500
    assert c.get(f"/api/tasks/{tid}").json()["storage"] == "hot"
    assert not (cold_dir / tid).exists()
    assert (data_dir / "published" / tid).read_bytes() == DATA


def test_restart_completes_hot_cleanup_after_crash(tmp_path):
    """Crash after routing switched cold but before hot deletion."""
    c, data_dir, cold_dir = make_cold_client(tmp_path)
    tid = publish(c, DATA)
    assert c.post(f"/api/tasks/{tid}/archive", json={}).json()["storage"] == "cold"

    # Rewind: simulate the crash window (cold copy verified+routed, hot not
    # yet deleted and not flagged).
    import sqlite3

    db = sqlite3.connect(data_dir / "uploader.db")
    db.execute("UPDATE tasks SET archive_deleted_hot = 0 WHERE id = ?", (tid,))
    db.commit()
    db.close()
    (data_dir / "published" / tid).write_bytes(DATA)  # hot copy still there

    # 重启后页面显示仍在冷卷，下载正常，重试完成清理
    c2 = TestClient(create_app(
        data_dir, ColdVolumes([storage_mod.ColdVolume("cold-a", cold_dir)])))
    st = c2.get(f"/api/tasks/{tid}").json()
    assert st["storage"] == "cold" and st["hot_deleted"] is False
    assert c2.get(f"/api/tasks/{tid}/download").content == DATA

    r = c2.post(f"/api/tasks/{tid}/archive", json={})
    assert r.status_code == 200 and r.json()["hot_deleted"] is True
    assert not (data_dir / "published" / tid).exists()
    assert (cold_dir / tid).read_bytes() == DATA
    assert c2.get(f"/api/tasks/{tid}/download").content == DATA


def test_restart_recovers_crash_between_record_and_routing_switch(tmp_path):
    """Crash after recording 'archived' but before switching routing."""
    import sqlite3

    c, data_dir, cold_dir = make_cold_client(tmp_path)
    tid = publish(c, DATA)
    assert c.post(f"/api/tasks/{tid}/archive", json={}).json()["storage"] == "cold"

    # Rewind that window: cold copy exists and is recorded, routing still hot.
    db = sqlite3.connect(data_dir / "uploader.db")
    db.execute(
        "UPDATE tasks SET storage = 'hot', archive_deleted_hot = 0 WHERE id = ?",
        (tid,))
    db.commit()
    db.close()
    (data_dir / "published" / tid).write_bytes(DATA)

    c2 = TestClient(create_app(
        data_dir, ColdVolumes([storage_mod.ColdVolume("cold-a", cold_dir)])))
    # Before the retry: download still routes to the intact hot copy, and the
    # task does not falsely claim to be on the cold volume.
    assert c2.get(f"/api/tasks/{tid}").json()["storage"] == "hot"
    assert c2.get(f"/api/tasks/{tid}/download").content == DATA

    r = c2.post(f"/api/tasks/{tid}/archive", json={})
    assert r.status_code == 200, r.text
    assert r.json()["storage"] == "cold" and r.json()["hot_deleted"] is True
    assert not (data_dir / "published" / tid).exists()
    assert (cold_dir / tid).read_bytes() == DATA
    assert c2.get(f"/api/tasks/{tid}").json()["storage"] == "cold"
    assert c2.get(f"/api/tasks/{tid}/download").content == DATA


def test_restart_cleans_stale_staging_from_interrupted_archive(tmp_path):
    """Crash mid-copy: staging file left behind, hot copy still authoritative."""
    import sqlite3

    c, data_dir, cold_dir = make_cold_client(tmp_path)
    tid = publish(c, DATA)

    # Simulate a process killed during the copy: DB says 'copying', a partial
    # staging file exists, routing is still hot.
    staging = cold_dir / ".staging" / f"{tid}.dead.tmp"
    staging.parent.mkdir(exist_ok=True)
    staging.write_bytes(b"partial")
    db = sqlite3.connect(data_dir / "uploader.db")
    db.execute(
        "UPDATE tasks SET archive_state = 'copying', archive_target = 'cold-a',"
        " archive_staging = ? WHERE id = ?", (str(staging), tid))
    db.commit()
    db.close()

    # 重启：原包仍可下载；归档重试成功，暂存残留被清理
    c2 = TestClient(create_app(
        data_dir, ColdVolumes([storage_mod.ColdVolume("cold-a", cold_dir)])))
    with c2:  # triggers lifespan startup cleanup
        st = c2.get(f"/api/tasks/{tid}").json()
        assert st["storage"] == "hot"
        assert st["archive_state"] is None  # no false success after restart
        assert not staging.exists()
        assert c2.get(f"/api/tasks/{tid}/download").content == DATA
        r = c2.post(f"/api/tasks/{tid}/archive", json={})
        assert r.status_code == 200 and r.json()["storage"] == "cold"
        assert c2.get(f"/api/tasks/{tid}/download").content == DATA


def test_download_streams_even_when_hot_file_is_unlinked(tmp_path, monkeypatch):
    """A transfer that opened the hot file before deletion still completes."""
    c, data_dir, cold_dir = make_cold_client(tmp_path)
    tid = publish(c, DATA)

    # Freeze streaming so the response body is produced lazily, after archive.
    real_delay = __import__("app.main", fromlist=["DOWNLOAD_DELAY_MS"])
    monkeypatch.setattr(real_delay, "DOWNLOAD_DELAY_MS", 5.0)

    with c.stream("GET", f"/api/tasks/{tid}/download") as r:
        assert r.status_code == 200
        it = r.iter_bytes()
        first = next(it)  # hot descriptor is open and the first chunk is read
        assert c.post(f"/api/tasks/{tid}/archive", json={}).status_code == 200
        assert not (data_dir / "published" / tid).exists()
        rest = b"".join(it)
    assert first + rest == DATA


def test_retry_replaces_unverified_leftover_at_cold_path(tmp_path):
    """A corrupt unrecorded file at the cold path must not block retries."""
    c, data_dir, cold_dir = make_cold_client(tmp_path)
    tid = publish(c, DATA)
    (cold_dir / tid).write_bytes(b"garbage from an old attempt")

    r = c.post(f"/api/tasks/{tid}/archive", json={})
    assert r.status_code == 200, r.text
    assert r.json()["storage"] == "cold" and r.json()["hot_deleted"] is True
    assert (cold_dir / tid).read_bytes() == DATA
    assert c.get(f"/api/tasks/{tid}/download").content == DATA


def test_archive_adopts_verified_cold_copy_race(tmp_path, monkeypatch):
    """A cold copy landed by a racing process is verified and adopted."""
    import sqlite3

    c, data_dir, cold_dir = make_cold_client(tmp_path)
    tid = publish(c, DATA)

    def copy_then_report(src, staging, expected_size, expected_sha256,
                         delay_ms=0.0):
        # Another process wins the race and lands the cold file + routing.
        (cold_dir / tid).write_bytes(DATA)
        db = sqlite3.connect(data_dir / "uploader.db")
        db.execute(
            "UPDATE tasks SET storage='cold', archive_target='cold-a',"
            " archive_state='archived' WHERE id = ?", (tid,))
        db.commit()
        db.close()
        Path(staging).write_bytes(DATA)
        Path(staging).unlink()

    monkeypatch.setattr(storage_mod, "copy_verified", copy_then_report)
    r = c.post(f"/api/tasks/{tid}/archive", json={})
    assert r.status_code == 200, r.text
    assert r.json()["hot_deleted"] is True
    assert (cold_dir / tid).read_bytes() == DATA
    assert not (data_dir / "published" / tid).exists()
    assert c.get(f"/api/tasks/{tid}").json()["storage"] == "cold"


def test_cold_volume_unavailable_reports_clearly(tmp_path):
    c, data_dir, cold_dir = make_cold_client(tmp_path)
    tid = publish(c, DATA)
    # Remove the cold volume directory entirely.
    import shutil

    shutil.rmtree(cold_dir)
    r = c.post(f"/api/tasks/{tid}/archive", json={"volume": "cold-a"})
    assert r.status_code == 503
    assert c.get(f"/api/tasks/{tid}").json()["storage"] == "hot"
    assert c.get(f"/api/tasks/{tid}/download").content == DATA


def test_storage_endpoint_lists_volumes(tmp_path):
    c, data_dir, cold_dir = make_cold_client(tmp_path)
    body = c.get("/api/storage").json()
    assert body["volumes"] == [{
        "name": "cold-a",
        "path": str(cold_dir.resolve()),
        "available": True,
        "free_bytes": body["volumes"][0]["free_bytes"],
    }]
    assert body["volumes"][0]["free_bytes"] > 0


def test_schema_migrates_old_database(tmp_path):
    """A DB created before cold archival is upgraded in place."""
    import sqlite3

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = sqlite3.connect(data_dir / "uploader.db")
    db.executescript("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY, filename TEXT NOT NULL, size INTEGER NOT NULL,
            sha256 TEXT NOT NULL, chunk_size INTEGER NOT NULL,
            total_chunks INTEGER NOT NULL,
            state TEXT NOT NULL DEFAULT 'uploading',
            created_at TEXT NOT NULL, published_at TEXT);
        CREATE TABLE chunks (
            task_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
            sha256 TEXT NOT NULL, size INTEGER NOT NULL,
            PRIMARY KEY (task_id, chunk_index));
    """)
    db.execute(
        "INSERT INTO tasks VALUES ('old','f',1,? ,1048576,1,'uploading',"
        "'2020-01-01','')", (sha256(b"x"),))
    db.commit()
    db.close()

    c = TestClient(create_app(data_dir))
    st = c.get("/api/tasks/old").json()
    assert st["storage"] == "hot" and st["archive_state"] is None


def test_empty_file_archive(tmp_path):
    c, data_dir, cold_dir = make_cold_client(tmp_path)
    tid = publish(c, b"")
    r = c.post(f"/api/tasks/{tid}/archive", json={})
    assert r.status_code == 200 and r.json()["storage"] == "cold"
    assert (cold_dir / tid).read_bytes() == b""
    assert c.get(f"/api/tasks/{tid}/download").content == b""
