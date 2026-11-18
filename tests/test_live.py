"""Live multi-process tests: two uvicorn processes share one data dir.

Covers: concurrent chunk uploads (incl. duplicate and conflicting content),
concurrent finalize from both processes, kill/restart mid-upload, and that
published packages remain downloadable after restarts.

Run:  .venv/bin/python -m pytest tests/test_live.py -v
"""
import hashlib
import os
import random
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1024 * 1024


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Server:
    def __init__(self, data_dir: Path, log_dir: Path, env_extra: dict | None = None):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        env = dict(os.environ, DATA_DIR=str(data_dir), PYTHONPATH=str(ROOT))
        env.update(env_extra or {})
        self.log = open(log_dir / f"server-{self.port}.log", "wb")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", str(self.port)],
            env=env, cwd=str(ROOT), stdout=self.log, stderr=subprocess.STDOUT,
        )

    def wait_ready(self, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server on {self.port} exited early")
            try:
                if httpx.get(self.base + "/api/health", timeout=1).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.2)
        raise TimeoutError(f"server on {self.port} not ready")

    def kill(self):
        self.proc.kill()
        self.proc.wait()
        self.log.close()

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.log.close()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def create_task(client: httpx.Client, base: str, data: bytes) -> str:
    r = client.post(f"{base}/api/tasks", json={
        "filename": "diag.tar.gz", "size": len(data), "sha256": sha256(data)})
    assert r.status_code == 201, r.text
    return r.json()["task_id"]


def put_chunk(client: httpx.Client, base: str, tid: str, i: int, payload: bytes):
    return client.put(f"{base}/api/tasks/{tid}/chunks/{i}", content=payload)


def test_two_processes_concurrent_upload_and_finalize(tmp_path):
    data_dir = tmp_path / "data"
    s1, s2 = Server(data_dir, tmp_path), Server(data_dir, tmp_path)
    try:
        s1.wait_ready()
        s2.wait_ready()
        data = random.Random(7).randbytes(5 * CHUNK + 999)  # 6 chunks
        chunks = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)]

        with httpx.Client(timeout=30) as client:
            tid = create_task(client, s1.base, data)

            # 两个进程并发上传：正常分块 + 同内容重复分块
            jobs = []
            for i, c in enumerate(chunks):
                jobs.append((s1.base if i % 2 == 0 else s2.base, i, c))
            # 同内容重传（打到另一个进程）
            for i in range(3):
                jobs.append((s2.base if i % 2 == 0 else s1.base, i, chunks[i]))

            def send(job):
                base, i, payload = job
                return put_chunk(client, base, tid, i, payload)

            with ThreadPoolExecutor(max_workers=9) as ex:
                results = list(ex.map(send, jobs))
            assert all(r.status_code == 200 for r in results), \
                [r.text for r in results]

            st = client.get(f"{s2.base}/api/tasks/{tid}").json()
            assert st["received_count"] == 6 and st["missing_chunks"] == []

            # 异内容重传：拒绝且保留原块
            evil = bytearray(chunks[0])
            evil[0] ^= 0xFF
            r = put_chunk(client, s2.base, tid, 0, bytes(evil))
            assert r.status_code == 409

            # 两个进程并发提交同一任务
            with ThreadPoolExecutor(max_workers=2) as ex:
                futs = [ex.submit(client.post, f"{b}/api/tasks/{tid}/finalize")
                        for b in (s1.base, s2.base)]
                finals = [f.result() for f in futs]
            assert all(r.status_code == 200 for r in finals), \
                [r.text for r in finals]
            assert all(r.json()["state"] == "published" for r in finals)

            # 两个进程下载到的成品一致且正确；磁盘上只有一份成品
            for base in (s1.base, s2.base):
                dl = client.get(f"{base}/api/tasks/{tid}/download")
                assert dl.status_code == 200 and dl.content == data
            published = list((data_dir / "published").iterdir())
            assert len(published) == 1
            assert published[0].read_bytes() == data
    finally:
        s1.stop()
        s2.stop()


def test_kill_restart_mid_upload_and_published_survives(tmp_path):
    data_dir = tmp_path / "data"
    data = random.Random(11).randbytes(3 * CHUNK + 1)  # 4 chunks
    chunks = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)]

    s1 = Server(data_dir, tmp_path)
    s1.wait_ready()
    with httpx.Client(timeout=30) as client:
        tid = create_task(client, s1.base, data)
        assert put_chunk(client, s1.base, tid, 0, chunks[0]).status_code == 200
        assert put_chunk(client, s1.base, tid, 1, chunks[1]).status_code == 200
    s1.kill()  # 模拟进程崩溃（kill -9）

    s2 = Server(data_dir, tmp_path)
    try:
        s2.wait_ready()
        with httpx.Client(timeout=30) as client:
            st = client.get(f"{s2.base}/api/tasks/{tid}").json()
            assert st["received_chunks"] == [0, 1]  # 已收分块保留
            assert put_chunk(client, s2.base, tid, 2, chunks[2]).status_code == 200
            assert put_chunk(client, s2.base, tid, 3, chunks[3]).status_code == 200
            r = client.post(f"{s2.base}/api/tasks/{tid}/finalize")
            assert r.status_code == 200 and r.json()["state"] == "published"
    finally:
        s2.kill()  # 发布后再崩溃

    s3 = Server(data_dir, tmp_path)
    try:
        s3.wait_ready()
        with httpx.Client(timeout=30) as client:
            dl = client.get(f"{s3.base}/api/tasks/{tid}/download")
            assert dl.status_code == 200 and dl.content == data  # 已发布仍可下载
    finally:
        s3.stop()


def test_concurrent_duplicate_same_chunk_across_processes(tmp_path):
    """Same chunk, same content, raced from two processes: all 200, one record."""
    data_dir = tmp_path / "data"
    s1, s2 = Server(data_dir, tmp_path), Server(data_dir, tmp_path)
    try:
        s1.wait_ready()
        s2.wait_ready()
        data = random.Random(3).randbytes(CHUNK)
        with httpx.Client(timeout=30) as client:
            tid = create_task(client, s1.base, data)
            with ThreadPoolExecutor(max_workers=2) as ex:
                r1 = ex.submit(put_chunk, client, s1.base, tid, 0, data)
                r2 = ex.submit(put_chunk, client, s2.base, tid, 0, data)
                results = [r1.result(), r2.result()]
            assert all(r.status_code == 200 for r in results)
            st = client.get(f"{s1.base}/api/tasks/{tid}").json()
            assert st["received_chunks"] == [0]
            assert client.post(f"{s2.base}/api/tasks/{tid}/finalize").json()[
                "state"] == "published"
            assert client.get(f"{s1.base}/api/tasks/{tid}/download").content == data
    finally:
        s1.stop()
        s2.stop()


# ------------------------------------------------------- cold archive (live)


def wait_until(predicate, timeout=10.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_two_processes_archive_while_downloads_in_flight(tmp_path):
    """Both processes archive the same package while downloads are running.

    Downloads started before the migration and during the migration must all
    yield the complete file; afterwards only one verified cold copy remains
    and the hot product is gone.
    """
    data_dir = tmp_path / "data"
    cold_dir = tmp_path / "cold" / "cold-a"
    cold_dir.mkdir(parents=True)
    env = {
        "COLD_VOLUMES": f"cold-a={cold_dir}",
        "GSB_DOWNLOAD_DELAY_MS": "25",
        "GSB_ARCHIVE_DELAY_MS": "10",
    }
    s1 = Server(data_dir, tmp_path, env)
    s2 = Server(data_dir, tmp_path, env)
    try:
        s1.wait_ready()
        s2.wait_ready()
        data = random.Random(21).randbytes(6 * CHUNK + 321)  # 7 chunks
        with httpx.Client(timeout=60) as client:
            tid = create_task(client, s1.base, data)
            chunks = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)]
            with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
                rs = list(ex.map(
                    lambda ic: put_chunk(client, s1.base if ic[0] % 2 else s2.base,
                                         tid, ic[0], ic[1]),
                    enumerate(chunks)))
            assert all(r.status_code == 200 for r in rs)
            with ThreadPoolExecutor(max_workers=2) as ex:
                rs = [f.result() for f in [
                    ex.submit(client.post, f"{s1.base}/api/tasks/{tid}/finalize"),
                    ex.submit(client.post, f"{s2.base}/api/tasks/{tid}/finalize"),
                ]]
            assert all(r.status_code == 200 for r in rs), [r.text for r in rs]

            got: dict[str, bytes] = {}

            def download(name, base, delay_before=0.0):
                if delay_before:
                    time.sleep(delay_before)
                with httpx.stream("GET", f"{base}/api/tasks/{tid}/download",
                                  timeout=60) as r:
                    assert r.status_code == 200
                    got[name] = b"".join(r.iter_bytes())

            # Two downloads open the hot file before the archive starts.
            before1 = ThreadPoolExecutor(max_workers=1).submit(
                download, "before-1", s1.base)
            before2 = ThreadPoolExecutor(max_workers=1).submit(
                download, "before-2", s2.base)
            time.sleep(0.12)  # both are mid-transfer

            # Both processes archive concurrently.
            with ThreadPoolExecutor(max_workers=2) as ex:
                arch = [f.result() for f in [
                    ex.submit(client.post, f"{s1.base}/api/tasks/{tid}/archive",
                              json={"volume": "cold-a"}),
                    ex.submit(client.post, f"{s2.base}/api/tasks/{tid}/archive",
                              json={"volume": "cold-a"}),
                ]]
            # One more download starts while the archive calls are finishing.
            download("during", s2.base)

            before1.result(timeout=60)
            before2.result(timeout=60)
            assert all(r.status_code == 200 for r in arch), [r.text for r in arch]
            assert all(r.json()["storage"] == "cold" for r in arch)
            assert all(r.json()["volume"] == "cold-a" for r in arch)

            # Every in-flight / mid-migration transfer got the whole file.
            assert got["before-1"] == data
            assert got["before-2"] == data
            assert got["during"] == data

            # Afterwards both processes route to the single verified cold copy.
            for base in (s1.base, s2.base):
                dl = client.get(f"{base}/api/tasks/{tid}/download")
                assert dl.status_code == 200 and dl.content == data
                assert dl.headers["x-package-volume"] == "cold-a"
            assert (cold_dir / tid).read_bytes() == data
            assert not (data_dir / "published" / tid).exists()
            assert list((cold_dir / ".staging").iterdir()) == []
            assert [p.name for p in cold_dir.iterdir() if not p.name.startswith(".")] \
                == [tid]
            st = client.get(f"{s2.base}/api/tasks/{tid}").json()
            assert st["storage"] == "cold" and st["hot_deleted"] is True

            # Uploading and publishing other packages keeps working.
            tid2 = create_task(client, s1.base, data)
            with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
                rs = list(ex.map(
                    lambda ic: put_chunk(client, s2.base, tid2, ic[0], ic[1]),
                    enumerate(chunks)))
            assert all(r.status_code == 200 for r in rs)
            assert client.post(f"{s1.base}/api/tasks/{tid2}/finalize").json()[
                "state"] == "published"
            assert client.get(f"{s2.base}/api/tasks/{tid2}/download").content == data
    finally:
        s1.stop()
        s2.stop()


def test_kill_mid_archive_restart_keeps_package_downloadable(tmp_path):
    """kill -9 while copying to cold: package stays hot/downloadable, retry works."""
    data_dir = tmp_path / "data"
    cold_dir = tmp_path / "cold" / "cold-a"
    cold_dir.mkdir(parents=True)
    env = {"COLD_VOLUMES": f"cold-a={cold_dir}", "GSB_ARCHIVE_DELAY_MS": "80"}
    data = random.Random(31).randbytes(10 * CHUNK + 7)  # 11 chunks -> slow copy

    s1 = Server(data_dir, tmp_path, env)
    s1.wait_ready()
    with httpx.Client(timeout=60) as client:
        tid = create_task(client, s1.base, data)
        chunks = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)]
        for i, c in enumerate(chunks):
            assert put_chunk(client, s1.base, tid, i, c).status_code == 200
        assert client.post(f"{s1.base}/api/tasks/{tid}/finalize").json()[
            "state"] == "published"

        # Fire archive in the background and kill the process mid-copy.
        import threading

        def fire():
            try:
                client.post(f"{s1.base}/api/tasks/{tid}/archive",
                            json={"volume": "cold-a"})
            except Exception:
                pass

        t = threading.Thread(target=fire, daemon=True)
        t.start()
        assert wait_until(lambda: list((cold_dir / ".staging").glob("*")) != [],
                          timeout=20), "archive never started copying"
        s1.kill()
        t.join(timeout=5)

    # The interrupted staging file is a partial copy and must not be used.
    staging = list((cold_dir / ".staging").iterdir())
    assert staging, "expected an interrupted staging file"

    s2 = Server(data_dir, tmp_path, {"COLD_VOLUMES": f"cold-a={cold_dir}"})
    try:
        s2.wait_ready()
        with httpx.Client(timeout=60) as client:
            st = client.get(f"{s2.base}/api/tasks/{tid}").json()
            # 不得误报归档成功：仍走热卷，且有可用原包
            assert st["storage"] == "hot"
            assert st["volume"] is None
            assert st["archive_state"] is None
            assert client.get(f"{s2.base}/api/tasks/{tid}/download").content == data
            assert not (cold_dir / tid).exists()  # 没有不完整的冷卷成品

            # 重试归档成功，暂存区清理干净，热卷成品删除
            r = client.post(f"{s2.base}/api/tasks/{tid}/archive",
                            json={"volume": "cold-a"})
            assert r.status_code == 200, r.text
            assert r.json()["storage"] == "cold" and r.json()["hot_deleted"]
            assert (cold_dir / tid).read_bytes() == data
            assert not (data_dir / "published" / tid).exists()
            assert list((cold_dir / ".staging").iterdir()) == []
            assert client.get(f"{s2.base}/api/tasks/{tid}/download").content == data
    finally:
        s2.stop()


def test_archive_failure_on_readonly_volume_then_retry(tmp_path):
    """Cold volume write failure: 5xx, hot copy downloadable, retry succeeds."""
    data_dir = tmp_path / "data"
    cold_dir = tmp_path / "cold" / "cold-a"
    cold_dir.mkdir(parents=True)
    data = random.Random(41).randbytes(2 * CHUNK + 11)
    env = {"COLD_VOLUMES": f"cold-a={cold_dir}"}

    s = Server(data_dir, tmp_path, env)
    s.wait_ready()
    try:
        with httpx.Client(timeout=60) as client:
            tid = create_task(client, s.base, data)
            for i, c in enumerate(
                    [data[j:j + CHUNK] for j in range(0, len(data), CHUNK)]):
                assert put_chunk(client, s.base, tid, i, c).status_code == 200
            assert client.post(f"{s.base}/api/tasks/{tid}/finalize").json()[
                "state"] == "published"

            cold_dir.chmod(0o500)  # cannot create .staging
            try:
                r = client.post(f"{s.base}/api/tasks/{tid}/archive",
                                json={"volume": "cold-a"})
                assert r.status_code == 503, r.text
            finally:
                cold_dir.chmod(0o755)

            st = client.get(f"{s.base}/api/tasks/{tid}").json()
            assert st["storage"] == "hot"
            assert st["archive_error"]
            assert client.get(f"{s.base}/api/tasks/{tid}/download").content == data

            r = client.post(f"{s.base}/api/tasks/{tid}/archive",
                            json={"volume": "cold-a"})
            assert r.status_code == 200 and r.json()["storage"] == "cold"
            assert client.get(f"{s.base}/api/tasks/{tid}/download").content == data
    finally:
        s.stop()
