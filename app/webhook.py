import hmac
import logging

from fastapi import APIRouter, Request, Response, status

from app.config import get_settings

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
