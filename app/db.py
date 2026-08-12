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
    # One row per inbound message that reached the reply pipeline: what was
    # retrieved, what intent fired, the assembled prompt, and per-stage
    # timings. Purely observational — never read by the reply path itself,
    # only by the /admin/traces API. history_window and retrieval_hits hold
    # JSON arrays (see app/trace.py) rather than normalized child tables,
    # since they're written once and always read back whole.
    """
    CREATE TABLE IF NOT EXISTS traces (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        sender_id            TEXT    NOT NULL,
        created_at           INTEGER NOT NULL,
        incoming_text        TEXT    NOT NULL,
        history_window       TEXT    NOT NULL,
        intent               TEXT    NOT NULL,
        intent_latency_ms    REAL,
        retrieval_hits       TEXT    NOT NULL,
        retrieval_latency_ms REAL,
        system_prompt        TEXT,
        reply                TEXT,
        reply_source         TEXT    NOT NULL,
        llm_latency_ms       REAL,
        total_latency_ms     REAL    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_traces_sender ON traces (sender_id, id)",
    "CREATE INDEX IF NOT EXISTS idx_traces_expiry ON traces (created_at)",
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
    # Reference-image URLs seen per sender, kept separately from `messages`
    # (which only ever gets a bracketed placeholder) so a quote request can
    # hand the artists real photos. Same retention window as messages.
    """
    CREATE TABLE IF NOT EXISTS attachments (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        sender_id  TEXT    NOT NULL,
        url        TEXT    NOT NULL,
        created_at INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_attachments_sender ON attachments (sender_id, id)",
    "CREATE INDEX IF NOT EXISTS idx_attachments_expiry ON attachments (created_at)",
    # One row per quote sent to the artists' Telegram chat. telegram_message_id
    # is the id of the summary message the artist is expected to reply to —
    # that's how app.telegram_webhook correlates their reply back to a
    # sender_id, since the two conversations happen on entirely different
    # channels.
    """
    CREATE TABLE IF NOT EXISTS quote_requests (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        sender_id           TEXT    NOT NULL,
        telegram_message_id INTEGER NOT NULL UNIQUE,
        status              TEXT    NOT NULL DEFAULT 'pending'
                                     CHECK (status IN ('pending', 'answered')),
        created_at          INTEGER NOT NULL,
        resolved_at         INTEGER
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_quote_requests_message"
    " ON quote_requests (telegram_message_id)",
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


def sweep_expired_traces(now: int, retention_days: int) -> int:
    """Delete traces older than the retention window. Returns rows deleted.

    Shares messages' retention window rather than a dedicated setting — traces
    are a debugging aid over the same conversation history, so there's no
    reason for them to outlive the messages they were captured from.
    """
    cutoff = now - retention_days * 86400
    with connect() as conn:
        cursor = conn.execute("DELETE FROM traces WHERE created_at < ?", (cutoff,))
        return cursor.rowcount


def sweep_expired_attachments(now: int, retention_days: int) -> int:
    """Delete attachments older than the retention window. Returns rows deleted.

    Shares messages' retention window — an attachment URL is customer content
    tied to a conversation, so it shouldn't outlive the messages around it.
    """
    cutoff = now - retention_days * 86400
    with connect() as conn:
        cursor = conn.execute("DELETE FROM attachments WHERE created_at < ?", (cutoff,))
        return cursor.rowcount


async def sweep_loop() -> None:
    """Run the expiry sweep on a fixed interval until cancelled."""
    settings = get_settings()
    while True:
        try:
            now = int(time.time())
            deleted = await asyncio.to_thread(
                sweep_expired, now, settings.history_retention_days
            )
            logger.info("Expiry sweep deleted %d messages", deleted)
            traces_deleted = await asyncio.to_thread(
                sweep_expired_traces, now, settings.history_retention_days
            )
            logger.info("Expiry sweep deleted %d traces", traces_deleted)
            attachments_deleted = await asyncio.to_thread(
                sweep_expired_attachments, now, settings.history_retention_days
            )
            logger.info("Expiry sweep deleted %d attachments", attachments_deleted)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Expiry sweep failed; retrying next interval")
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
