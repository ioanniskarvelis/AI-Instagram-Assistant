import hashlib
import hmac
import logging

from fastapi import APIRouter, Request, Response, status
from pydantic import ValidationError

from app.config import get_settings
from app.history import Turn, append, recent
from app.instagram import MAX_MESSAGE_BYTES, send_text
from app.llm import generate_reply
from app.schemas import WebhookPayload

logger = logging.getLogger(__name__)

router = APIRouter()


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

    try:
        payload = WebhookPayload.model_validate_json(raw_body)
    except ValidationError as exc:
        logger.warning("Malformed webhook payload: %s", exc)
        return Response(status_code=status.HTTP_400_BAD_REQUEST)

    for sender_id, text in payload.replyable_messages():
        logger.info("Replying to %s (received %d chars)", sender_id, len(text))

        await append(sender_id, "user", text)

        window = await recent(sender_id, settings.history_window_messages)
        if not window:
            # Storage is unavailable; answer the message in front of us rather
            # than going silent.
            window = [Turn(role="user", text=text)]

        reply = await generate_reply(window)
        if reply is not None:
            reply_bytes = len(reply.encode("utf-8"))
            if reply_bytes > MAX_MESSAGE_BYTES:
                # The Graph API rejects over-long text outright. Truncating
                # would cut a customer off mid-sentence, so the canned
                # acknowledgement is the better degradation.
                logger.warning(
                    "Generated reply for %s was %d bytes, over the %d byte limit",
                    sender_id,
                    reply_bytes,
                    MAX_MESSAGE_BYTES,
                )
                reply = None
        if reply is None:
            reply = settings.canned_reply

        if await send_text(sender_id, reply):
            await append(sender_id, "assistant", reply)

    return Response(status_code=status.HTTP_200_OK)
