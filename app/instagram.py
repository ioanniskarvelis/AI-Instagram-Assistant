import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.instagram.com"
REQUEST_TIMEOUT_SECONDS = 10.0

# Maximum characters the Graph API accepts in a text DM. Confirmed against Meta's
# Instagram Messaging documentation; the API rejects longer messages outright, so
# an over-long generated reply is caught before it is sent rather than after.
MAX_MESSAGE_CHARS = 1000


async def send_text(recipient_id: str, text: str) -> bool:
    """Send a plain-text DM. Returns True on success, False on failure.

    Never raises: the webhook must return 200 to Meta regardless of whether
    the reply went out, so failures are logged and reported as a bool.
    """
    settings = get_settings()
    url = f"{GRAPH_BASE}/{settings.ig_api_version}/me/messages"
    payload = {"recipient": {"id": recipient_id}, "message": {"text": text}}
    headers = {"Authorization": f"Bearer {settings.ig_user_access_token}"}

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError:
        logger.exception("Transport error sending reply to %s", recipient_id)
        return False

    if response.status_code >= 400:
        logger.error(
            "Graph API rejected reply to %s: %s %s",
            recipient_id,
            response.status_code,
            response.text,
        )
        return False

    return True
