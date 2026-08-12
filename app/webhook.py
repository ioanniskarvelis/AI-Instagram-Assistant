import asyncio
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, TypeVar

from fastapi import APIRouter, Request, Response, status
from pydantic import ValidationError

from app import admin_store, quotes, telegram, trace
from app.config import get_settings
from app.history import Turn, append, recent
from app.instagram import MAX_MESSAGE_BYTES, send_text
from app.intent import Intent, classify
from app.llm import build_system_prompt, generate_reply
from app.rag import retrieve_scored
from app.schemas import WebhookPayload

logger = logging.getLogger(__name__)

router = APIRouter()

_T = TypeVar("_T")


async def _timed(awaitable: Awaitable[_T]) -> tuple[_T, float]:
    """Run `awaitable`, returning its result alongside elapsed time in ms."""
    start = time.perf_counter()
    result = await awaitable
    return result, (time.perf_counter() - start) * 1000


@dataclass
class _PendingBurst:
    """A debounce timer in flight for one sender, plus the messages it has
    buffered so far. Held in-memory only — fine for the single uvicorn
    process this app runs as, but a burst in flight is lost on restart."""

    task: "asyncio.Task[None]"
    texts: list[str] = field(default_factory=list)


# sender_id -> the burst currently waiting out its debounce window.
_pending: dict[str, _PendingBurst] = {}


def _buffer_message(sender_id: str, text: str) -> None:
    """Add `text` to the sender's in-flight burst and restart its timer.

    Cancelling the old task and starting a new one is how "restart the timer
    on every new message" is implemented: the old task's asyncio.sleep raises
    CancelledError and it exits without replying, leaving this newer task as
    the one that will eventually fire.
    """
    existing = _pending.get(sender_id)
    if existing is not None:
        existing.task.cancel()
        texts = existing.texts
    else:
        texts = []
    texts.append(text)

    task = asyncio.create_task(_debounced_reply(sender_id))
    _pending[sender_id] = _PendingBurst(task=task, texts=texts)


async def _debounced_reply(sender_id: str) -> None:
    """Wait out the debounce window, then reply to the sender's buffered
    burst as a single exchange. Never raises: this runs detached from the
    request that scheduled it, so an uncaught exception would otherwise only
    surface in asyncio's "Task exception was never retrieved" log.
    """
    settings = get_settings()
    try:
        await asyncio.sleep(settings.reply_debounce_seconds)
    except asyncio.CancelledError:
        # A newer message superseded this task; it already scheduled its own.
        return

    try:
        await _reply_to_burst(sender_id)
    except Exception:
        logger.exception("Debounced reply pipeline failed for %s", sender_id)
    finally:
        current = _pending.get(sender_id)
        if current is not None and current.task is asyncio.current_task():
            del _pending[sender_id]


async def _reply_to_burst(sender_id: str) -> None:
    """Generate and send one reply for everything buffered since the last
    debounce fired. Admin kill switches are checked here, at fire time, not
    when each message arrived, so a mid-wait disable still takes effect.
    """
    burst = _pending.get(sender_id)
    if burst is None:
        return

    if await asyncio.to_thread(admin_store.get_bot_disabled):
        logger.info("Not replying to %s: bot is globally disabled", sender_id)
        return

    if await asyncio.to_thread(admin_store.get_conversation_disabled, sender_id):
        logger.info("Not replying to %s: conversation is disabled", sender_id)
        return

    settings = get_settings()
    text = "\n".join(burst.texts)
    logger.info(
        "Replying to %s (%d buffered message(s), %d chars)",
        sender_id,
        len(burst.texts),
        len(text),
    )
    total_start = time.perf_counter()

    window = await recent(sender_id, settings.history_window_messages)
    if not window:
        # Storage is unavailable; answer the message(s) in front of us rather
        # than going silent.
        window = [Turn(role="user", text=text)]

    # retrieve_scored() and classify() never raise; each degrades
    # independently (hits to [], intent to Intent.GENERAL) on any
    # internal failure.
    (scored_hits, retrieval_ms), (msg_intent, intent_ms) = await asyncio.gather(
        _timed(retrieve_scored(text, settings.rag_top_k)),
        _timed(classify(window)),
    )
    examples = [hit.example for hit in scored_hits]
    retrieval_hits = [
        trace.RetrievalHit(question=h.example.question, reply=h.example.reply, score=h.score)
        for h in scored_hits
    ]

    system_prompt: str | None = None
    llm_ms: float | None = None

    if msg_intent is Intent.COMPLAINT:
        logger.info(
            "Auto-disabling conversation %s: classified as complaint", sender_id
        )
        await asyncio.to_thread(
            admin_store.set_conversation_disabled, sender_id, True
        )
        reply, reply_source = None, "suppressed"
    else:
        system_prompt = build_system_prompt(examples, msg_intent)
        result, llm_ms = await _timed(generate_reply(window, examples, msg_intent))
        reply = result.text
        reply_source = "generated"
        if reply is not None:
            reply_bytes = len(reply.encode("utf-8"))
            if reply_bytes > MAX_MESSAGE_BYTES:
                # The Graph API rejects over-long text outright.
                # Truncating would cut a customer off mid-sentence, so
                # the canned acknowledgement is the better degradation.
                logger.warning(
                    "Generated reply for %s was %d bytes, over the %d byte limit",
                    sender_id,
                    reply_bytes,
                    MAX_MESSAGE_BYTES,
                )
                reply = None
        if reply is None:
            reply_source = "canned"
            reply = settings.canned_reply

        if result.quote_summary:
            await _request_quote(sender_id, result.quote_summary)

    await trace.save(
        trace.TraceRecord(
            sender_id=sender_id,
            incoming_text=text,
            history_window=[trace.HistoryTurn(role=t.role, text=t.text) for t in window],
            intent=msg_intent.value,
            intent_latency_ms=intent_ms,
            retrieval_hits=retrieval_hits,
            retrieval_latency_ms=retrieval_ms,
            system_prompt=system_prompt,
            reply=reply,
            reply_source=reply_source,
            llm_latency_ms=llm_ms,
            total_latency_ms=(time.perf_counter() - total_start) * 1000,
        )
    )

    if reply_source == "suppressed":
        return

    if await send_text(sender_id, reply):
        await append(sender_id, "assistant", reply)


async def _request_quote(sender_id: str, summary: str) -> None:
    """Send the gathered details plus every reference image on file for this
    sender to the artists' Telegram chat, and remember which Telegram
    message they need to reply to for the price to reach the customer.
    """
    image_urls = await quotes.image_urls_for(sender_id)
    telegram_message_id = await telegram.send_quote_request(sender_id, summary, image_urls)
    if telegram_message_id is None:
        logger.warning("Quote request for %s did not reach Telegram", sender_id)
        return
    await quotes.create_quote_request(sender_id, telegram_message_id)


@router.get("/webhook")
async def verify(request: Request) -> Response:
    """Answer Meta's subscription handshake."""
    settings = get_settings()
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge", "")

    if (
        mode == "subscribe"
        and token is not None
        and hmac.compare_digest(token, settings.ig_verify_token)
    ):
        return Response(content=challenge, media_type="text/plain")

    logger.warning("Webhook verification failed (mode=%s)", mode)
    return Response(status_code=status.HTTP_403_FORBIDDEN)


def _signature_valid(raw_body: bytes, header: str | None, app_secret: str) -> bool:
    """Check Meta's X-Hub-Signature-256 against the raw request body."""
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.removeprefix("sha256="))


@router.post("/webhook")
async def receive(request: Request) -> Response:
    """Receive DM events and reply.

    The signature is checked against the raw body before parsing. Validating a
    re-serialised body would compare a different byte sequence than Meta signed,
    producing a check that silently always fails or, worse, is trivially bypassed.
    """
    settings = get_settings()
    raw_body = await request.body()

    if not _signature_valid(
        raw_body, request.headers.get("X-Hub-Signature-256"), settings.ig_app_secret
    ):
        logger.warning("Rejected webhook delivery with invalid signature")
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    logger.debug("Webhook body: %s", raw_body.decode("utf-8", errors="replace"))

    try:
        payload = WebhookPayload.model_validate_json(raw_body)
    except ValidationError as exc:
        logger.warning("Malformed webhook payload: %s", exc)
        return Response(status_code=status.HTTP_400_BAD_REQUEST)

    # Recorded unconditionally, like the messages below — if a quote later
    # gets requested, every reference image the sender has ever sent should
    # be on hand for it regardless of when the allowlist/kill-switch checks
    # below let a reply through.
    for sender_id, url in payload.image_urls(settings.ig_account_id):
        await quotes.record_attachment(sender_id, url)

    allowed_senders = settings.allowed_sender_id_set
    for sender_id, text in payload.replyable_messages(settings.ig_account_id):
        await append(sender_id, "user", text)

        if allowed_senders and sender_id not in allowed_senders:
            # The assistant isn't live for everyone yet. Store the message so
            # nothing is lost, but don't generate or send a reply.
            logger.info("Ignoring %s: not in the allowed sender list", sender_id)
            continue

        # Buffer and (re)start this sender's debounce timer rather than
        # replying inline: a burst of quick messages should settle into one
        # reply, not one per message. The actual pipeline runs later, in the
        # background, once `reply_debounce_seconds` pass without a new one.
        _buffer_message(sender_id, text)

    return Response(status_code=status.HTTP_200_OK)
