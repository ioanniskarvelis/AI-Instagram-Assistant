"""Storage for the Telegram quoting flow: reference-image URLs seen per
sender, and the correlation between a quote sent to the artists and the
sender_id it's for.

Mirrors the pattern in history.py/trace.py: plain sqlite3 functions run via
asyncio.to_thread, since a sqlite3 connection is bound to its creating
thread. Every write here is best-effort and never raises — losing an
attachment URL or a quote-request row must not stop a reply going out.
"""

import asyncio
import logging
import time

from app import db

logger = logging.getLogger(__name__)


def _insert_attachment(sender_id: str, url: str) -> None:
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO attachments (sender_id, url, created_at) VALUES (?, ?, ?)",
            (sender_id, url, int(time.time())),
        )


async def record_attachment(sender_id: str, url: str) -> bool:
    """Store one reference-image URL. Returns False on failure; never raises."""
    try:
        await asyncio.to_thread(_insert_attachment, sender_id, url)
        return True
    except Exception:
        logger.exception("Failed to store attachment for %s", sender_id)
        return False


def _select_image_urls(sender_id: str, limit: int) -> list[str]:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT url FROM attachments WHERE sender_id = ?"
            " ORDER BY id DESC LIMIT ?",
            (sender_id, limit),
        ).fetchall()
    return [row[0] for row in reversed(rows)]


async def image_urls_for(sender_id: str, limit: int = 10) -> list[str]:
    """Return up to `limit` of the sender's most recent reference images,
    oldest first. Returns [] on failure."""
    try:
        return await asyncio.to_thread(_select_image_urls, sender_id, limit)
    except Exception:
        logger.exception("Failed to read attachments for %s", sender_id)
        return []


def _insert_quote_request(sender_id: str, telegram_message_id: int) -> None:
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO quote_requests"
            " (sender_id, telegram_message_id, status, created_at)"
            " VALUES (?, ?, 'pending', ?)",
            (sender_id, telegram_message_id, int(time.time())),
        )


async def create_quote_request(sender_id: str, telegram_message_id: int) -> bool:
    """Record that `telegram_message_id` is the summary message the artists
    should reply to for `sender_id`. Returns False on failure; never raises."""
    try:
        await asyncio.to_thread(_insert_quote_request, sender_id, telegram_message_id)
        return True
    except Exception:
        logger.exception("Failed to record quote request for %s", sender_id)
        return False


def _resolve_quote_request(telegram_message_id: int) -> str | None:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT sender_id FROM quote_requests"
            " WHERE telegram_message_id = ? AND status = 'pending'",
            (telegram_message_id,),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE quote_requests SET status = 'answered', resolved_at = ?"
            " WHERE telegram_message_id = ?",
            (int(time.time()), telegram_message_id),
        )
    return row[0]


async def resolve_quote_request(telegram_message_id: int) -> str | None:
    """Mark the pending request at `telegram_message_id` answered and return
    its sender_id, or None if it isn't a known pending request (already
    answered, unrelated Telegram reply, or a lookup failure — never raises).
    Marking it answered here, in the same statement that reads it, means a
    duplicate Telegram delivery of the same reply resolves it at most once.
    """
    try:
        return await asyncio.to_thread(_resolve_quote_request, telegram_message_id)
    except Exception:
        logger.exception("Failed to resolve quote request %s", telegram_message_id)
        return None
