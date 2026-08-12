"""Synchronous SQLite queries backing the admin API.

Mirrors the pattern in db.py/history.py: plain sqlite3 functions that callers
run via asyncio.to_thread, since a sqlite3 connection is bound to its
creating thread.
"""

import time
from dataclasses import dataclass

from app import db


@dataclass(frozen=True)
class ConversationSummary:
    sender_id: str
    bot_disabled: bool
    created_at: int
    updated_at: int
    message_count: int
    last_message_preview: str


@dataclass(frozen=True)
class MessageDetail:
    id: int
    role: str
    text: str
    created_at: int


def get_bot_disabled() -> bool:
    with db.connect() as conn:
        row = conn.execute("SELECT disabled FROM bot_state WHERE id = 1").fetchone()
    return bool(row and row[0])


def set_bot_disabled(disabled: bool) -> None:
    with db.connect() as conn:
        conn.execute(
            "UPDATE bot_state SET disabled = ? WHERE id = 1", (int(disabled),)
        )


def get_conversation_disabled(sender_id: str) -> bool:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT disabled FROM conversation_state WHERE sender_id = ?",
            (sender_id,),
        ).fetchone()
    return bool(row and row[0])


def set_conversation_disabled(sender_id: str, disabled: bool) -> None:
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO conversation_state (sender_id, disabled) VALUES (?, ?)"
            " ON CONFLICT (sender_id) DO UPDATE SET disabled = excluded.disabled",
            (sender_id, int(disabled)),
        )


def list_conversations(limit: int, offset: int) -> list[ConversationSummary]:
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT sender_id,
                   MIN(created_at) AS created_at,
                   MAX(created_at) AS updated_at,
                   COUNT(*)        AS message_count
            FROM messages
            GROUP BY sender_id
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

        disabled_senders = {
            row[0]
            for row in conn.execute(
                "SELECT sender_id FROM conversation_state WHERE disabled = 1"
            ).fetchall()
        }

        summaries = []
        for sender_id, created_at, updated_at, message_count in rows:
            preview_row = conn.execute(
                "SELECT text FROM messages WHERE sender_id = ? ORDER BY id DESC LIMIT 1",
                (sender_id,),
            ).fetchone()
            summaries.append(
                ConversationSummary(
                    sender_id=sender_id,
                    bot_disabled=sender_id in disabled_senders,
                    created_at=created_at,
                    updated_at=updated_at,
                    message_count=message_count,
                    last_message_preview=preview_row[0] if preview_row else "",
                )
            )
    return summaries


def get_conversation_messages(sender_id: str) -> list[MessageDetail]:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, role, text, created_at FROM messages"
            " WHERE sender_id = ? ORDER BY id ASC",
            (sender_id,),
        ).fetchall()
    return [
        MessageDetail(id=id_, role=role, text=text, created_at=created_at)
        for id_, role, text, created_at in rows
    ]


def delete_conversation_history(sender_id: str) -> int:
    with db.connect() as conn:
        cursor = conn.execute("DELETE FROM messages WHERE sender_id = ?", (sender_id,))
        return cursor.rowcount


def last_inbound_message_at() -> int | None:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT MAX(created_at) FROM messages WHERE role = 'user'"
        ).fetchone()
    return row[0] if row and row[0] is not None else None


def database_ping() -> float:
    """Open a connection and run a trivial query. Returns latency in ms."""
    start = time.monotonic()
    with db.connect() as conn:
        conn.execute("SELECT 1")
    return (time.monotonic() - start) * 1000
