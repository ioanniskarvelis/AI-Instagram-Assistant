"""Outbound calls to the artists' Telegram chat for the quoting flow.

Reference-image URLs are passed straight through to Telegram rather than
downloaded and re-uploaded — Telegram's servers fetch the URL themselves for
sendPhoto/sendMediaGroup, which works because Instagram's attachment CDN
links are public, time-limited URLs, not ones requiring our page token.
"""

import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
REQUEST_TIMEOUT_SECONDS = 15.0

# Telegram rejects a media group outside 2-10 items; a single photo goes
# through sendPhoto instead (see _send_photos).
MAX_MEDIA_GROUP_PHOTOS = 10


async def _send_photos(client: httpx.AsyncClient, base: str, chat_id: str, urls: list[str]) -> bool:
    photos = urls[:MAX_MEDIA_GROUP_PHOTOS]
    if len(urls) > MAX_MEDIA_GROUP_PHOTOS:
        logger.warning(
            "Quote request carries %d reference images; only sending the first %d",
            len(urls), MAX_MEDIA_GROUP_PHOTOS,
        )

    if len(photos) == 1:
        response = await client.post(
            f"{base}/sendPhoto", json={"chat_id": chat_id, "photo": photos[0]}
        )
    else:
        media = [{"type": "photo", "media": url} for url in photos]
        response = await client.post(
            f"{base}/sendMediaGroup", json={"chat_id": chat_id, "media": media}
        )

    if response.status_code >= 400:
        logger.error("Telegram rejected reference photo(s): %s %s", response.status_code, response.text)
        return False
    return True


async def send_quote_request(sender_id: str, summary: str, image_urls: list[str]) -> int | None:
    """Send a quote request to the studio's Telegram chat.

    Returns the message_id of the summary text message — the id the artists
    are expected to reply to — or None if Telegram isn't configured or the
    send failed. Never raises.
    """
    settings = get_settings()
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.debug("Telegram not configured; skipping quote request for %s", sender_id)
        return None

    base = f"{API_BASE}/bot{settings.telegram_bot_token}"
    text = (
        f"New quote request (Instagram sender {sender_id}):\n\n{summary}\n\n"
        "Reply directly to this message with the price."
    )

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            if image_urls and not await _send_photos(client, base, settings.telegram_chat_id, image_urls):
                return None

            response = await client.post(
                f"{base}/sendMessage",
                json={"chat_id": settings.telegram_chat_id, "text": text},
            )
    except httpx.HTTPError:
        logger.exception("Transport error sending quote request for %s", sender_id)
        return None

    if response.status_code >= 400:
        logger.error(
            "Telegram rejected quote request for %s: %s %s",
            sender_id, response.status_code, response.text,
        )
        return None

    return response.json()["result"]["message_id"]
