import asyncio
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import get_settings

logger = logging.getLogger(__name__)

SWEEP_INTERVAL_SECONDS = 6 * 60 * 60

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS messages (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        sender_id  TEXT    NOT NULL,
        role       TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
        text       TEXT    NOT NULL,
        created_at INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages (sender_id, id)",
    "CREATE INDEX IF NOT EXISTS idx_messages_expiry ON messages (created_at)",
    # Single-row table holding the global kill switch. The row is seeded by
    # init_schema so callers can always UPDATE it rather than upserting.
    """
    CREATE TABLE IF NOT EXISTS bot_state (
        id       INTEGER PRIMARY KEY CHECK (id = 1),
        disabled INTEGER NOT NULL DEFAULT 0
    )
    """,
    # Per-sender kill switch. Only present for senders an admin has muted;
    # absence means enabled, so no row is needed for the common case.
    """
    CREATE TABLE IF NOT EXISTS conversation_state (
        sender_id TEXT PRIMARY KEY,
        disabled  INTEGER NOT NULL DEFAULT 0
    )
    """,
    "INSERT OR IGNORE INTO bot_state (id, disabled) VALUES (1, 0)",
)


def _database_path() -> Path:
    return Path(get_settings().db_path)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Open a connection for a single operation.

    A connection is opened per operation rather than shared process-wide because
    every call runs on an arbitrary `asyncio.to_thread` worker, and a sqlite3
    connection is bound to its creating thread. Opening an existing file is cheap
    enough at this volume to buy that whole class of bug out of the design.
    """
    conn = sqlite3.connect(_database_path())
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema() -> None:
    """Create the database file, table and indexes. Safe to call repeatedly."""
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        # journal_mode is persisted in the database file, so it is set once here
        # rather than on every connection.
        conn.execute("PRAGMA journal_mode=WAL")
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)


def sweep_expired(now: int, retention_days: int) -> int:
    """Delete messages older than the retention window. Returns rows deleted."""
    cutoff = now - retention_days * 86400
    with connect() as conn:
        cursor = conn.execute("DELETE FROM messages WHERE created_at < ?", (cutoff,))
        return cursor.rowcount


async def sweep_loop() -> None:
    """Run the expiry sweep on a fixed interval until cancelled."""
    settings = get_settings()
    while True:
        try:
            deleted = await asyncio.to_thread(
                sweep_expired, int(time.time()), settings.history_retention_days
            )
            logger.info("Expiry sweep deleted %d messages", deleted)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Expiry sweep failed; retrying next interval")
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
