import json

import httpx
import respx

BASE = "https://api.telegram.org/botTEST_TOKEN"


@respx.mock
async def test_send_quote_request_returns_none_when_not_configured(env, monkeypatch):
    from app.config import get_settings
    from app.telegram import send_quote_request

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    get_settings.cache_clear()

    assert await send_quote_request("SENDER_1", "summary", []) is None


@respx.mock
async def test_send_quote_request_without_images_sends_one_text_message(env, monkeypatch):
    from app.config import get_settings
    from app.telegram import send_quote_request

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TEST_TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    get_settings.cache_clear()

    route = respx.post(f"{BASE}/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 55}})
    )

    message_id = await send_quote_request("SENDER_1", "small rose, forearm, 5cm", [])

    assert message_id == 55
    body = json.loads(route.calls.last.request.content)
    assert body["chat_id"] == "-100123"
    assert "small rose, forearm, 5cm" in body["text"]
    assert "SENDER_1" in body["text"]


@respx.mock
async def test_send_quote_request_with_one_image_uses_send_photo(env, monkeypatch):
    from app.config import get_settings
    from app.telegram import send_quote_request

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TEST_TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    get_settings.cache_clear()

    photo_route = respx.post(f"{BASE}/sendPhoto").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})
    )
    message_route = respx.post(f"{BASE}/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 56}})
    )

    message_id = await send_quote_request("SENDER_1", "summary", ["https://cdn/a.jpg"])

    assert message_id == 56
    assert photo_route.called
    photo_body = json.loads(photo_route.calls.last.request.content)
    assert photo_body == {"chat_id": "-100123", "photo": "https://cdn/a.jpg"}
    assert message_route.called


@respx.mock
async def test_send_quote_request_with_multiple_images_uses_media_group(env, monkeypatch):
    from app.config import get_settings
    from app.telegram import send_quote_request

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TEST_TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    get_settings.cache_clear()

    media_route = respx.post(f"{BASE}/sendMediaGroup").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": []})
    )
    respx.post(f"{BASE}/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 57}})
    )

    urls = ["https://cdn/a.jpg", "https://cdn/b.jpg"]
    message_id = await send_quote_request("SENDER_1", "summary", urls)

    assert message_id == 57
    media_body = json.loads(media_route.calls.last.request.content)
    assert media_body["media"] == [
        {"type": "photo", "media": "https://cdn/a.jpg"},
        {"type": "photo", "media": "https://cdn/b.jpg"},
    ]


@respx.mock
async def test_send_quote_request_returns_none_when_photo_send_fails(env, monkeypatch):
    from app.config import get_settings
    from app.telegram import send_quote_request

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TEST_TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    get_settings.cache_clear()

    respx.post(f"{BASE}/sendPhoto").mock(return_value=httpx.Response(400, json={"ok": False}))
    message_route = respx.post(f"{BASE}/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})
    )

    result = await send_quote_request("SENDER_1", "summary", ["https://cdn/a.jpg"])

    assert result is None
    # A failed photo send must not still fire off the text message.
    assert not message_route.called


@respx.mock
async def test_send_quote_request_returns_none_when_text_send_fails(env, monkeypatch):
    from app.config import get_settings
    from app.telegram import send_quote_request

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TEST_TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    get_settings.cache_clear()

    respx.post(f"{BASE}/sendMessage").mock(return_value=httpx.Response(500, json={"ok": False}))

    assert await send_quote_request("SENDER_1", "summary", []) is None


@respx.mock
async def test_send_quote_request_returns_none_on_transport_error(env, monkeypatch):
    from app.config import get_settings
    from app.telegram import send_quote_request

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TEST_TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    get_settings.cache_clear()

    respx.post(f"{BASE}/sendMessage").mock(side_effect=httpx.ConnectError("boom"))

    assert await send_quote_request("SENDER_1", "summary", []) is None
