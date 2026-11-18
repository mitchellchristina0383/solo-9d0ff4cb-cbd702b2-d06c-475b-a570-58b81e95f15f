"""SQLite access helpers.

WAL mode allows concurrent readers with a single writer across processes;
busy_timeout makes the loser of a write race wait instead of failing.
Connections run in autocommit mode (isolation_level=None); every state
mutation is a single statement, hence atomic.

Cold-archive columns (added by migration to databases created before cold
archival existed):

  storage            'hot' (published/) or 'cold' (a cold volume). Download
                     routing is decided by this column only.
  archive_state      NULL while not archiving, 'copying' once a cold copy is
                     being written, 'archived' once a verified cold copy
                     exists. 'archived' is recorded BEFORE the hot file is
                     deleted, and the hot file is only deleted after
                     `storage` has been switched to 'cold'.
  archive_target     name of the cold volume the copy lives on.
  archive_source     hot path the cold copy was made from (diagnostics).
  archive_staging    staging path of an in-flight copy.
  archive_error      last failed archive attempt (user-visible diagnostic).
  archive_deleted_hot 1 once the hot copy has been removed.
  archived_at        timestamp of the first durable 'archived' record.
"""
from __future__ import annotations

import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    size          INTEGER NOT NULL,
    sha256        TEXT NOT NULL,
    chunk_size    INTEGER NOT NULL,
    total_chunks  INTEGER NOT NULL,
    state         TEXT NOT NULL DEFAULT 'uploading'
                  CHECK (state IN ('uploading', 'published', 'failed')),
    created_at    TEXT NOT NULL,
    published_at  TEXT,
    storage       TEXT NOT NULL DEFAULT 'hot'
                  CHECK (storage IN ('hot', 'cold')),
    archive_state TEXT
                  CHECK (archive_state IN ('copying', 'archived')),
    archive_target TEXT,
    archive_source TEXT,
    archive_staging TEXT,
    archive_error TEXT,
    archive_deleted_hot INTEGER NOT NULL DEFAULT 0,
    archived_at   TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    task_id     TEXT NOT NULL REFERENCES tasks(id),
    chunk_index INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    size        INTEGER NOT NULL,
    PRIMARY KEY (task_id, chunk_index)
);
"""

# Indexes are created only after migration, so a pre-existing database whose
# `tasks` table lacks the archive columns is upgraded first.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);
CREATE INDEX IF NOT EXISTS idx_tasks_archive_state ON tasks(archive_state);
"""

# Columns introduced by cold archival. Databases created by older versions of
# the uploader are upgraded in place by adding whichever ones are missing.
MIGRATION_COLUMNS = {
    "storage": "TEXT NOT NULL DEFAULT 'hot'",
    "archive_state": "TEXT",
    "archive_target": "TEXT",
    "archive_source": "TEXT",
    "archive_staging": "TEXT",
    "archive_error": "TEXT",
    "archive_deleted_hot": "INTEGER NOT NULL DEFAULT 0",
    "archived_at": "TEXT",
}


def connect(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(db_path), timeout=30.0, isolation_level=None, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    # busy_timeout must be in place before any statement that may need an
    # exclusive lock (notably switching the journal mode on a fresh DB).
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    _retry(lambda: conn.execute("PRAGMA journal_mode=WAL"))
    return conn


def _retry(op, attempts: int = 50, delay: float = 0.1):
    """Run *op*, retrying while SQLite reports the database as locked.

    Processes starting at the same time (e.g. uvicorn --workers N) can trip
    over the brief exclusive lock taken when a fresh database switches to WAL.
    """
    for attempt in range(attempts):
        try:
            return op()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == attempts - 1:
                raise
            time.sleep(delay)


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    for column, decl in MIGRATION_COLUMNS.items():
        if column not in existing:
            try:
                conn.execute(f"ALTER TABLE tasks ADD COLUMN {column} {decl}")
            except sqlite3.OperationalError as exc:
                # Another process migrating the same database won the race.
                if "duplicate column" not in str(exc).lower():
                    raise


def init_db(db_path) -> None:
    conn = connect(db_path)
    try:
        _retry(lambda: conn.executescript(SCHEMA))
        _retry(lambda: _migrate(conn))
        _retry(lambda: conn.executescript(INDEXES))
    finally:
        conn.close()
