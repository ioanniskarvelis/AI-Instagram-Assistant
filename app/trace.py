"""Per-message observability trace: what the pipeline saw and did.

Captures, for every inbound message that reaches the reply pipeline, the
history window it was classified/generated against, the retrieved RAG
examples with their similarity scores, the intent that fired, the exact
system prompt sent to the model, the reply, and per-stage latencies. Written
by app.webhook after each message; read back by the /admin/traces API for the
trace viewer.

Mirrors the pattern in db.py/history.py: plain sqlite3 functions run via
asyncio.to_thread, since a sqlite3 connection is bound to its creating
thread. Writing is best-effort and never raises — a trace is a debugging
aid, not something that should ever affect whether a customer gets a reply.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from app import db

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoryTurn:
    role: str
    text: str


@dataclass(frozen=True)
class RetrievalHit:
    question: str
    reply: str
    score: float


@dataclass(frozen=True)
class TraceRecord:
    """One row's worth of data to persist. `id`/`created_at` are assigned on
    write, so they aren't part of this input shape."""

    sender_id: str
    incoming_text: str
    history_window: list[HistoryTurn]
    intent: str
    intent_latency_ms: float | None
    retrieval_hits: list[RetrievalHit]
    retrieval_latency_ms: float | None
    system_prompt: str | None
    reply: str | None
    reply_source: str  # "generated" | "canned" | "suppressed"
    llm_latency_ms: float | None
    total_latency_ms: float


@dataclass(frozen=True)
class TraceSummary:
    id: int
    sender_id: str
    created_at: int
    incoming_text: str
    intent: str
    reply_source: str
    reply: str | None
    total_latency_ms: float


@dataclass(frozen=True)
class TraceDetail(TraceSummary):
    history_window: list[HistoryTurn]
    intent_latency_ms: float | None
    retrieval_hits: list[RetrievalHit]
    retrieval_latency_ms: float | None
    system_prompt: str | None
    llm_latency_ms: float | None


def _insert(record: TraceRecord) -> None:
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO traces (
                sender_id, created_at, incoming_text, history_window, intent,
                intent_latency_ms, retrieval_hits, retrieval_latency_ms,
                system_prompt, reply, reply_source, llm_latency_ms,
                total_latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.sender_id,
                int(time.time()),
                record.incoming_text,
                json.dumps([{"role": t.role, "text": t.text} for t in record.history_window]),
                record.intent,
                record.intent_latency_ms,
                json.dumps(
                    [
                        {"question": h.question, "reply": h.reply, "score": h.score}
                        for h in record.retrieval_hits
                    ]
                ),
                record.retrieval_latency_ms,
                record.system_prompt,
                record.reply,
                record.reply_source,
                record.llm_latency_ms,
                record.total_latency_ms,
            ),
        )


async def save(record: TraceRecord) -> bool:
    """Persist one trace. Returns False on failure; never raises."""
    try:
        await asyncio.to_thread(_insert, record)
        return True
    except Exception:
        logger.exception("Failed to store trace for %s", record.sender_id)
        return False


_SUMMARY_COLUMNS = (
    "id, sender_id, created_at, incoming_text, intent, reply_source, reply,"
    " total_latency_ms"
)


def _row_to_summary(row: tuple) -> TraceSummary:
    return TraceSummary(
        id=row[0],
        sender_id=row[1],
        created_at=row[2],
        incoming_text=row[3],
        intent=row[4],
        reply_source=row[5],
        reply=row[6],
        total_latency_ms=row[7],
    )


def list_traces(
    limit: int,
    offset: int,
    sender_id: str | None = None,
    intent: str | None = None,
) -> list[TraceSummary]:
    clauses = []
    params: list[object] = []
    if sender_id is not None:
        clauses.append("sender_id = ?")
        params.append(sender_id)
    if intent is not None:
        clauses.append("intent = ?")
        params.append(intent)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT {_SUMMARY_COLUMNS} FROM traces{where}"
            " ORDER BY id DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
    return [_row_to_summary(row) for row in rows]


def get_trace(trace_id: int) -> TraceDetail | None:
    with db.connect() as conn:
        row = conn.execute(
            """
            SELECT id, sender_id, created_at, incoming_text, intent,
                   reply_source, reply, total_latency_ms, history_window,
                   intent_latency_ms, retrieval_hits, retrieval_latency_ms,
                   system_prompt, llm_latency_ms
            FROM traces WHERE id = ?
            """,
            (trace_id,),
        ).fetchone()
    if row is None:
        return None

    summary = _row_to_summary(row[:8])
    history_window = [HistoryTurn(**t) for t in json.loads(row[8])]
    retrieval_hits = [RetrievalHit(**h) for h in json.loads(row[10])]
    return TraceDetail(
        **summary.__dict__,
        history_window=history_window,
        intent_latency_ms=row[9],
        retrieval_hits=retrieval_hits,
        retrieval_latency_ms=row[11],
        system_prompt=row[12],
        llm_latency_ms=row[13],
    )
