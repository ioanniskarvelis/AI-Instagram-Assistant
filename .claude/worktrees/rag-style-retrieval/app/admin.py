"""Admin API consumed by the studio website's assistant dashboard.

Every route requires a bearer token matching ASSISTANT_ADMIN_KEY. The token
is shared with the website's Netlify proxy, never with the browser directly
— see 210tattoo-website/netlify/functions/assistant-proxy.ts, which checks
Supabase admin auth before forwarding here with this key attached.
"""

import asyncio
import hmac
import logging
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from app import admin_store
from app.config import get_settings

logger = logging.getLogger(__name__)

ComponentStatus = Literal["ok", "degraded", "error"]


def require_admin(request: Request) -> None:
    settings = get_settings()
    if not settings.assistant_admin_key:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Admin API not configured")

    auth_header = request.headers.get("authorization", "")
    token = auth_header.removeprefix("Bearer ").strip() if auth_header else ""
    if not token or not hmac.compare_digest(token, settings.assistant_admin_key):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid admin token")


router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


# ── Schemas ──────────────────────────────────────────────────────────────


class HealthComponent(BaseModel):
    status: ComponentStatus
    latency_ms: float | None = None
    detail: str | None = None
    last_received: str | None = None


class HealthResponse(BaseModel):
    status: ComponentStatus
    components: dict[str, HealthComponent]


class BotStatus(BaseModel):
    globally_disabled: bool


class OkResponse(BaseModel):
    ok: bool = True


class ConversationSummary(BaseModel):
    sender_id: str
    bot_disabled: bool
    created_at: str | None
    updated_at: str | None
    message_count: int
    last_message_preview: str


class MessageDetail(BaseModel):
    id: int
    role: str
    content: str
    timestamp: str | None


class ConversationDetail(ConversationSummary):
    messages: list[MessageDetail]


# ── Helpers ──────────────────────────────────────────────────────────────


def _iso(epoch_seconds: int | None) -> str | None:
    if epoch_seconds is None:
        return None
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def _to_summary(row: admin_store.ConversationSummary) -> ConversationSummary:
    return ConversationSummary(
        sender_id=row.sender_id,
        bot_disabled=row.bot_disabled,
        created_at=_iso(row.created_at),
        updated_at=_iso(row.updated_at),
        message_count=row.message_count,
        last_message_preview=row.last_message_preview,
    )


# ── Routes ───────────────────────────────────────────────────────────────


@router.get("/health")
async def health() -> HealthResponse:
    settings = get_settings()

    try:
        latency_ms = await asyncio.to_thread(admin_store.database_ping)
        database = HealthComponent(status="ok", latency_ms=round(latency_ms, 1))
    except Exception as exc:
        logger.exception("Admin health check: database ping failed")
        database = HealthComponent(status="error", detail=str(exc))

    anthropic = HealthComponent(
        status="ok" if settings.anthropic_api_key else "error",
        detail=None if settings.anthropic_api_key else "ANTHROPIC_API_KEY not set",
    )

    try:
        last_received = await asyncio.to_thread(admin_store.last_inbound_message_at)
        instagram_webhook = HealthComponent(
            status="ok",
            last_received=_iso(last_received) if last_received else "never",
        )
    except Exception as exc:
        logger.exception("Admin health check: last-message lookup failed")
        instagram_webhook = HealthComponent(status="error", detail=str(exc))

    components = {
        "database": database,
        "anthropic": anthropic,
        "instagram_webhook": instagram_webhook,
    }
    statuses = {c.status for c in components.values()}
    overall: ComponentStatus = (
        "error" if "error" in statuses else "degraded" if "degraded" in statuses else "ok"
    )
    return HealthResponse(status=overall, components=components)


@router.get("/bot/status")
async def bot_status() -> BotStatus:
    disabled = await asyncio.to_thread(admin_store.get_bot_disabled)
    return BotStatus(globally_disabled=disabled)


@router.post("/bot/disable")
async def bot_disable() -> OkResponse:
    await asyncio.to_thread(admin_store.set_bot_disabled, True)
    return OkResponse()


@router.post("/bot/enable")
async def bot_enable() -> OkResponse:
    await asyncio.to_thread(admin_store.set_bot_disabled, False)
    return OkResponse()


@router.get("/conversations")
async def conversations(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[ConversationSummary]:
    rows = await asyncio.to_thread(admin_store.list_conversations, limit, offset)
    return [_to_summary(row) for row in rows]


@router.get("/conversations/{sender_id}")
async def conversation_detail(sender_id: str) -> ConversationDetail:
    messages = await asyncio.to_thread(admin_store.get_conversation_messages, sender_id)
    if not messages:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No conversation for this sender")

    disabled = await asyncio.to_thread(admin_store.get_conversation_disabled, sender_id)
    summary = ConversationSummary(
        sender_id=sender_id,
        bot_disabled=disabled,
        created_at=_iso(messages[0].created_at),
        updated_at=_iso(messages[-1].created_at),
        message_count=len(messages),
        last_message_preview=messages[-1].text,
    )
    return ConversationDetail(
        **summary.model_dump(),
        messages=[
            MessageDetail(
                id=m.id, role=m.role, content=m.text, timestamp=_iso(m.created_at)
            )
            for m in messages
        ],
    )


@router.post("/conversations/{sender_id}/disable")
async def conversation_disable(sender_id: str) -> OkResponse:
    await asyncio.to_thread(admin_store.set_conversation_disabled, sender_id, True)
    return OkResponse()


@router.post("/conversations/{sender_id}/enable")
async def conversation_enable(sender_id: str) -> OkResponse:
    await asyncio.to_thread(admin_store.set_conversation_disabled, sender_id, False)
    return OkResponse()


@router.delete("/conversations/{sender_id}/history")
async def conversation_reset(sender_id: str) -> OkResponse:
    await asyncio.to_thread(admin_store.delete_conversation_history, sender_id)
    return OkResponse()
