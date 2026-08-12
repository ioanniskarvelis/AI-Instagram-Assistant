import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.instagram.com"
REQUEST_TIMEOUT_SECONDS = 10.0

# Maximum UTF-8 encoded bytes the Graph API accepts in a text DM. Meta's
# Instagram Messaging documentation states message text "must be UTF-8 and be
# 1000 bytes or less" (developers.facebook.com/docs/messenger-platform/
# instagram/features/send-message). The limit is bytes, not characters — a
# 1000-character Greek reply is ~2000 bytes — so the guard must measure the
# encoded length, or the API rejects a message that looked fine locally.
MAX_MESSAGE_BYTES = 1000


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
