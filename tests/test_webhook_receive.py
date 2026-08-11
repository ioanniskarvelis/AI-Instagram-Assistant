import json

import httpx
import respx
from fastapi.testclient import TestClient

from tests.conftest import CANNED_REPLY, sign

ENDPOINT = "https://graph.instagram.com/v26.0/me/messages"
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"

GENERATED = "Καλησπέρα! Πες μου περισσότερα για το σχέδιο."


def _anthropic_reply(text: str = GENERATED) -> dict:
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


def _mock_llm(text: str = GENERATED):
    return respx.post(ANTHROPIC_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_anthropic_reply(text))
    )


def _body(message: dict) -> bytes:
    payload = {
        "object": "instagram",
        "entry": [
            {
                "id": "STUDIO",
                "time": 1723000000,
                "messaging": [
                    {
                        "sender": {"id": "SENDER_1"},
                        "recipient": {"id": "STUDIO"},
                        "timestamp": 1723000000,
                        "message": message,
                    }
                ],
            }
        ],
    }
    return json.dumps(payload).encode()


def _post(client, body: bytes, signature: str | None = None):
    headers = {"Content-Type": "application/json"}
    headers["X-Hub-Signature-256"] = sign(body) if signature is None else signature
    return client.post("/webhook", content=body, headers=headers)


def _stored(sender_id: str) -> list[tuple[str, str]]:
    from app import db

    with db.connect() as conn:
        return conn.execute(
            "SELECT role, text FROM messages WHERE sender_id = ? ORDER BY id",
            (sender_id,),
        ).fetchall()


@respx.mock
def test_text_message_triggers_one_generated_reply(client):
    _mock_llm()
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    response = _post(client, _body({"mid": "m1", "text": "Γεια σας"}))

    assert response.status_code == 200
    assert route.call_count == 1

    sent = json.loads(route.calls.last.request.content)
    assert sent["recipient"]["id"] == "SENDER_1"
    assert sent["message"]["text"] == GENERATED


@respx.mock
def test_both_turns_are_stored(client):
    _mock_llm()
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    _post(client, _body({"mid": "m1", "text": "Γεια σας"}))

    assert _stored("SENDER_1") == [
        ("user", "Γεια σας"),
        ("assistant", GENERATED),
    ]


@respx.mock
def test_second_message_carries_the_first_exchange(client):
    llm = _mock_llm()
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    _post(client, _body({"mid": "m1", "text": "πρώτο"}))
    _post(client, _body({"mid": "m2", "text": "δεύτερο"}))

    body = json.loads(llm.calls.last.request.content)
    assert body["messages"] == [
        {"role": "user", "content": "πρώτο"},
        {"role": "assistant", "content": GENERATED},
        {"role": "user", "content": "δεύτερο"},
    ]


@respx.mock
def test_generation_failure_falls_back_to_canned_reply(client):
    respx.post(ANTHROPIC_ENDPOINT).mock(
        return_value=httpx.Response(
            400,
            json={"type": "error", "error": {"type": "invalid_request_error",
                                             "message": "bad"}},
        )
    )
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    response = _post(client, _body({"mid": "m1", "text": "Γεια σας"}))

    assert response.status_code == 200
    sent = json.loads(route.calls.last.request.content)
    assert sent["message"]["text"] == CANNED_REPLY


@respx.mock
def test_overlong_generated_reply_falls_back_to_canned(client):
    from app.instagram import MAX_MESSAGE_BYTES

    # "ω" encodes to 2 bytes in UTF-8. This length is chosen to exceed the
    # byte limit while staying under the character count of the byte limit,
    # so this reply would sail past a character-counting guard (it's fewer
    # than MAX_MESSAGE_BYTES characters) and only trips a byte-counting one.
    overlong = "ω" * (MAX_MESSAGE_BYTES // 2 + 1)
    assert len(overlong) < MAX_MESSAGE_BYTES
    assert len(overlong.encode("utf-8")) > MAX_MESSAGE_BYTES

    _mock_llm(overlong)
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    response = _post(client, _body({"mid": "m1", "text": "Γεια σας"}))

    assert response.status_code == 200
    sent = json.loads(route.calls.last.request.content)
    assert sent["message"]["text"] == CANNED_REPLY
    # The rejected reply must not enter history either.
    assert _stored("SENDER_1") == [
        ("user", "Γεια σας"),
        ("assistant", CANNED_REPLY),
    ]


@respx.mock
def test_send_failure_stores_no_assistant_turn(client):
    _mock_llm()
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(500, json={"error": "server error"})
    )

    response = _post(client, _body({"mid": "m1", "text": "Γεια σας"}))

    assert response.status_code == 200
    assert _stored("SENDER_1") == [("user", "Γεια σας")]


@respx.mock
def test_echo_message_stores_nothing_and_sends_nothing(client):
    llm = _mock_llm()
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={}))

    body = _body({"mid": "m2", "text": "our own reply", "is_echo": True})
    response = _post(client, body)

    assert response.status_code == 200
    assert route.call_count == 0
    assert llm.call_count == 0
    assert _stored("SENDER_1") == []


@respx.mock
def test_reply_suppressed_for_sender_outside_allowlist(env, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("ALLOWED_SENDER_IDS", "SOME_OTHER_SENDER")
    get_settings.cache_clear()

    from app.main import create_app

    _mock_llm()
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    with TestClient(create_app()) as client:
        response = _post(client, _body({"mid": "m1", "text": "Γεια σας"}))

    assert response.status_code == 200
    assert route.call_count == 0
    assert _stored("SENDER_1") == [("user", "Γεια σας")]


@respx.mock
def test_reply_sent_for_sender_inside_allowlist(env, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("ALLOWED_SENDER_IDS", "OTHER_SENDER,SENDER_1")
    get_settings.cache_clear()

    from app.main import create_app

    _mock_llm()
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    with TestClient(create_app()) as client:
        response = _post(client, _body({"mid": "m1", "text": "Γεια σας"}))

    assert response.status_code == 200
    assert route.call_count == 1
    assert _stored("SENDER_1") == [("user", "Γεια σας"), ("assistant", GENERATED)]


@respx.mock
def test_invalid_signature_is_rejected(client):
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={}))

    body = _body({"mid": "m3", "text": "hello"})
    response = _post(client, body, signature="sha256=deadbeef")

    assert response.status_code == 403
    assert route.call_count == 0


@respx.mock
def test_missing_signature_header_is_rejected(client):
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={}))

    body = _body({"mid": "m4", "text": "hello"})
    response = client.post(
        "/webhook", content=body, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 403
    assert route.call_count == 0


@respx.mock
def test_malformed_payload_returns_400(client):
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={}))

    body = json.dumps({"not": "a webhook"}).encode()
    response = _post(client, body)

    assert response.status_code == 400
    assert route.call_count == 0


@respx.mock
def test_send_failure_still_returns_200(client):
    _mock_llm()
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(500, json={"error": "server error"})
    )

    response = _post(client, _body({"mid": "m5", "text": "hello"}))

    assert response.status_code == 200
