FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV DATA_DIR=/data
# Cold archival target: a separate mounted volume (docker-compose mounts
# cold-data at /cold). Empty means "no cold volume configured"; uploads,
# publishing and downloads still work, only archival is unavailable.
ENV COLD_VOLUMES=cold-local=/cold
VOLUME ["/data", "/cold"]
EXPOSE 8000

# 两个 worker 进程共享同一 SQLite(WAL) 与数据目录，
# 用于验证多进程并发上传/提交的一致性设计。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
