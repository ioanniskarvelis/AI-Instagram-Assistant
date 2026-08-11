import json

import httpx
import respx

from tests.conftest import CANNED_REPLY, sign

ENDPOINT = "https://graph.instagram.com/v22.0/me/messages"


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


@respx.mock
def test_text_message_triggers_one_reply(client):
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    body = _body({"mid": "m1", "text": "Γεια σας"})
    response = _post(client, body)

    assert response.status_code == 200
    assert route.call_count == 1

    sent = json.loads(route.calls.last.request.content)
    assert sent["recipient"]["id"] == "SENDER_1"
    assert sent["message"]["text"] == CANNED_REPLY


@respx.mock
def test_echo_message_triggers_no_reply(client):
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={}))

    body = _body({"mid": "m2", "text": "our own reply", "is_echo": True})
    response = _post(client, body)

    assert response.status_code == 200
    assert route.call_count == 0


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
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(500, json={"error": "server error"})
    )

    body = _body({"mid": "m5", "text": "hello"})
    response = _post(client, body)

    assert response.status_code == 200
