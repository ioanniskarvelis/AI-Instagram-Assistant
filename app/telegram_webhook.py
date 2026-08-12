"""Receives the artists' price replies from Telegram and relays them to the
customer on Instagram.

An artist replies (Telegram's native reply-to) directly to the quote-request
message app.telegram sent. That reply_to_message.message_id is how this
correlates the reply back to a sender_id — the two conversations happen on
entirely different channels, so there's no shared conversation state to key
off otherwise. See app.quotes for the pending-request bookkeeping.
"""

import hmac
import logging

from fastapi import APIRouter, Request, Response, status

from app import quotes
from app.config import get_settings
from app.history import append, recent
from app.instagram import send_text
from app.llm import generate_quote_announcement

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/telegram-webhook")
async def receive(request: Request) -> Response:
    settings = get_settings()

    if not settings.telegram_webhook_secret:
        logger.warning("Rejected Telegram webhook: TELEGRAM_WEBHOOK_SECRET not configured")
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    provided = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if not provided or not hmac.compare_digest(provided, settings.telegram_webhook_secret):
        logger.warning("Rejected Telegram webhook: bad or missing secret token")
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    try:
        update = await request.json()
    except ValueError:
        return Response(status_code=status.HTTP_400_BAD_REQUEST)

    message = update.get("message") or {}

    # Only the configured studio chat can resolve a quote — anyone who adds
    # the bot elsewhere shouldn't be able to trigger an Instagram send.
    if str(message.get("chat", {}).get("id", "")) != str(settings.telegram_chat_id):
        return Response(status_code=status.HTTP_200_OK)

    reply_to = message.get("reply_to_message")
    quote_text = message.get("text")
    if not reply_to or not quote_text:
        # Not a reply to our quote-request message — ordinary chat, ignore.
        return Response(status_code=status.HTTP_200_OK)

    telegram_message_id = reply_to.get("message_id")
    if telegram_message_id is None:
        return Response(status_code=status.HTTP_200_OK)

    sender_id = await quotes.resolve_quote_request(telegram_message_id)
    if sender_id is None:
        # Already resolved, or a reply to some other Telegram message.
        return Response(status_code=status.HTTP_200_OK)

    window = await recent(sender_id, settings.history_window_messages)
    reply = await generate_quote_announcement(window, quote_text)
    if reply is None:
        # Phrasing failed — relay the artist's own words rather than leaving
        # the customer with no answer at all.
        reply = quote_text

    if await send_text(sender_id, reply):
        await append(sender_id, "assistant", reply)

    return Response(status_code=status.HTTP_200_OK)
