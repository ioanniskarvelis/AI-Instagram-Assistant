import json

import httpx
import respx

from tests.conftest import ACCESS_TOKEN

ENDPOINT = "https://graph.instagram.com/v22.0/me/messages"

# respx patches httpcore, which the app's httpx.AsyncClient goes through.
# TestClient's ASGITransport bypasses httpcore entirely, so requests to the
# app under test are never intercepted — only outbound Graph API calls are.


@respx.mock
async def test_send_text_posts_expected_payload():
    from app.instagram import send_text

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    assert await send_text("SENDER_1", "hello") is True
    assert route.called

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body == {
        "recipient": {"id": "SENDER_1"},
        "message": {"text": "hello"},
    }
    assert request.headers["Authorization"] == f"Bearer {ACCESS_TOKEN}"


@respx.mock
async def test_send_text_returns_false_on_api_error():
    from app.instagram import send_text

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad token"}})
    )

    assert await send_text("SENDER_1", "hello") is False


@respx.mock
async def test_send_text_returns_false_on_transport_error():
    from app.instagram import send_text

    respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("boom"))

    assert await send_text("SENDER_1", "hello") is False
