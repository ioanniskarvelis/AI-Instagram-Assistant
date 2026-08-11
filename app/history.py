import asyncio
import logging
import time
from dataclasses import dataclass

from app import db

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Turn:
    """One side of one exchange. A domain type, not a wire type."""

    role: str
    text: str


def _insert(sender_id: str, role: str, text: str) -> None:
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO messages (sender_id, role, text, created_at)"
            " VALUES (?, ?, ?, ?)",
            (sender_id, role, text, int(time.time())),
        )


def _select_recent(sender_id: str, limit: int) -> list[Turn]:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT role, text FROM messages WHERE sender_id = ?"
            " ORDER BY id DESC LIMIT ?",
            (sender_id, limit),
        ).fetchall()

    turns = [Turn(role=role, text=text) for role, text in reversed(rows)]

    # A sliding window can begin mid-exchange, but the API requires the first
    # message to be from the user — a leading assistant turn is a 400, not a
    # soft failure.
    while turns and turns[0].role != "user":
        turns.pop(0)
    return turns


async def append(sender_id: str, role: str, text: str) -> bool:
    """Store one turn. Returns False on failure; never raises.

    A storage failure must not stop the customer getting a reply, so the caller
    is told about it rather than interrupted by it.
    """
    try:
        await asyncio.to_thread(_insert, sender_id, role, text)
        return True
    except Exception:
        logger.exception("Failed to store %s turn for %s", role, sender_id)
        return False


async def recent(sender_id: str, limit: int) -> list[Turn]:
    """Return the last `limit` turns, oldest first. Returns [] on failure."""
    try:
        return await asyncio.to_thread(_select_recent, sender_id, limit)
    except Exception:
        logger.exception("Failed to read history for %s", sender_id)
        return []
