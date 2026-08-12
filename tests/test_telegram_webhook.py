import json

import httpx
import respx
from fastapi.testclient import TestClient

INSTAGRAM_ENDPOINT = "https://graph.instagram.com/v26.0/me/messages"
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
SECRET = "test-telegram-secret"


def _configured_client(monkeypatch):
    from app.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TEST_TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", SECRET)
    get_settings.cache_clear()

    return TestClient(create_app())


def _update(text: str, reply_to_message_id: int | None = 42, chat_id: str = "-100123") -> dict:
    message = {
        "message_id": 999,
        "chat": {"id": int(chat_id)},
        "text": text,
    }
    if reply_to_message_id is not None:
        message["reply_to_message"] = {"message_id": reply_to_message_id}
    return {"update_id": 1, "message": message}


def _post(client, body: dict, secret: str | None = SECRET):
    headers = {}
    if secret is not None:
        headers["X-Telegram-Bot-Api-Secret-Token"] = secret
    return client.post("/telegram-webhook", json=body, headers=headers)


def _anthropic_text(text: str) -> dict:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def test_missing_secret_header_is_rejected(env, monkeypatch):
    with _configured_client(monkeypatch) as client:
        response = _post(client, _update("150"), secret=None)
    assert response.status_code == 403


def test_wrong_secret_header_is_rejected(env, monkeypatch):
    with _configured_client(monkeypatch) as client:
        response = _post(client, _update("150"), secret="wrong")
    assert response.status_code == 403


def test_webhook_disabled_when_secret_not_configured(env, monkeypatch):
    from app.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "")
    get_settings.cache_clear()

    with TestClient(create_app()) as client:
        response = client.post(
            "/telegram-webhook",
            json=_update("150"),
            headers={"X-Telegram-Bot-Api-Secret-Token": "anything"},
        )
    assert response.status_code == 403


def test_reply_from_a_different_chat_is_ignored(env, monkeypatch):
    with _configured_client(monkeypatch) as client:
        response = _post(client, _update("150", chat_id="-999999"))
    assert response.status_code == 200


def test_message_that_is_not_a_reply_is_ignored(env, monkeypatch):
    with _configured_client(monkeypatch) as client:
        response = _post(client, _update("150", reply_to_message_id=None))
    assert response.status_code == 200


def test_reply_to_an_unknown_message_is_a_no_op(env, monkeypatch):
    with _configured_client(monkeypatch) as client:
        response = _post(client, _update("150", reply_to_message_id=42))
    assert response.status_code == 200


@respx.mock
def test_reply_to_a_pending_quote_announces_the_price_and_stores_it(env, monkeypatch):
    from app import quotes

    with _configured_client(monkeypatch) as client:
        quotes._insert_quote_request("SENDER_1", 42)  # seed a pending request

        respx.post(ANTHROPIC_ENDPOINT).mock(
            return_value=httpx.Response(200, json=_anthropic_text("Η τιμή είναι 150€! 🐼"))
        )
        route = respx.post(INSTAGRAM_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"message_id": "mid.1"})
        )

        response = _post(client, _update("150"))

        assert response.status_code == 200
        assert route.call_count == 1
        sent = json.loads(route.calls.last.request.content)
        assert sent["recipient"]["id"] == "SENDER_1"
        assert sent["message"]["text"] == "Η τιμή είναι 150€! 🐼"

        from app import db
        with db.connect() as conn:
            stored = conn.execute(
                "SELECT role, text FROM messages WHERE sender_id = 'SENDER_1' ORDER BY id"
            ).fetchall()
        assert stored == [("assistant", "Η τιμή είναι 150€! 🐼")]


@respx.mock
def test_duplicate_reply_to_the_same_quote_is_a_no_op(env, monkeypatch):
    from app import quotes

    with _configured_client(monkeypatch) as client:
        quotes._insert_quote_request("SENDER_1", 42)

        respx.post(ANTHROPIC_ENDPOINT).mock(
            return_value=httpx.Response(200, json=_anthropic_text("Η τιμή είναι 150€! 🐼"))
        )
        route = respx.post(INSTAGRAM_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"message_id": "mid.1"})
        )

        _post(client, _update("150"))
        second = _post(client, _update("150"))

        assert second.status_code == 200
        assert route.call_count == 1


@respx.mock
def test_announcement_failure_relays_the_raw_quote_text(env, monkeypatch):
    from app import quotes

    with _configured_client(monkeypatch) as client:
        quotes._insert_quote_request("SENDER_1", 42)

        respx.post(ANTHROPIC_ENDPOINT).mock(
            return_value=httpx.Response(
                400,
                json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}},
            )
        )
        route = respx.post(INSTAGRAM_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"message_id": "mid.1"})
        )

        _post(client, _update("180-200"))

        sent = json.loads(route.calls.last.request.content)
        assert sent["message"]["text"] == "180-200"
