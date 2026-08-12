import asyncio
import json

import httpx
import respx

from app.history import append
from tests.conftest import ADMIN_KEY, CANNED_REPLY, sign

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"

AUTH = {"Authorization": f"Bearer {ADMIN_KEY}"}


def _seed(sender_id: str, *turns: tuple[str, str]) -> None:
    for role, text in turns:
        asyncio.run(append(sender_id, role, text))


# ── Auth ─────────────────────────────────────────────────────────────────


def test_admin_route_requires_auth(client):
    resp = client.get("/admin/bot/status")
    assert resp.status_code == 401


def test_admin_route_rejects_wrong_token(client):
    resp = client.get("/admin/bot/status", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_admin_route_accepts_valid_token(client):
    resp = client.get("/admin/bot/status", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json() == {"globally_disabled": False}


# ── Health ───────────────────────────────────────────────────────────────


def test_health_reports_ok_components(client):
    resp = client.get("/admin/health", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["components"]["database"]["status"] == "ok"
    assert body["components"]["anthropic"]["status"] == "ok"
    assert body["components"]["instagram_webhook"]["last_received"] == "never"


def test_health_reflects_last_inbound_message(client):
    _seed("SENDER_1", ("user", "hi"))
    resp = client.get("/admin/health", headers=AUTH)
    assert resp.json()["components"]["instagram_webhook"]["last_received"] != "never"


# ── Global kill switch ───────────────────────────────────────────────────


def test_disable_and_enable_bot(client):
    assert client.post("/admin/bot/disable", headers=AUTH).json() == {"ok": True}
    assert client.get("/admin/bot/status", headers=AUTH).json() == {
        "globally_disabled": True
    }

    assert client.post("/admin/bot/enable", headers=AUTH).json() == {"ok": True}
    assert client.get("/admin/bot/status", headers=AUTH).json() == {
        "globally_disabled": False
    }


def _body(text: str) -> bytes:
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
                        "message": {"mid": "m1", "text": text},
                    }
                ],
            }
        ],
    }
    return json.dumps(payload).encode()


def test_globally_disabled_bot_stores_but_does_not_reply(client):
    client.post("/admin/bot/disable", headers=AUTH)

    body = _body("hello")
    with respx.mock:
        send_route = respx.post(
            "https://graph.instagram.com/v26.0/me/messages"
        ).mock(return_value=httpx.Response(200, json={}))
        resp = client.post(
            "/webhook",
            content=body,
            headers={"Content-Type": "application/json", "X-Hub-Signature-256": sign(body)},
        )
        assert resp.status_code == 200
        assert not send_route.called

    detail = client.get("/admin/conversations/SENDER_1", headers=AUTH).json()
    assert detail["message_count"] == 1
    assert detail["messages"][0]["role"] == "user"


def test_conversation_disable_blocks_only_that_sender(client):
    client.post("/admin/conversations/SENDER_1/disable", headers=AUTH)

    body = _body("hello")
    with respx.mock:
        send_route = respx.post(
            "https://graph.instagram.com/v26.0/me/messages"
        ).mock(return_value=httpx.Response(200, json={}))
        client.post(
            "/webhook",
            content=body,
            headers={"Content-Type": "application/json", "X-Hub-Signature-256": sign(body)},
        )
        assert not send_route.called


# ── Conversations ────────────────────────────────────────────────────────


def test_list_conversations(client):
    _seed("SENDER_1", ("user", "hi"), ("assistant", CANNED_REPLY))
    _seed("SENDER_2", ("user", "yo"))

    resp = client.get("/admin/conversations", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert {c["sender_id"] for c in body} == {"SENDER_1", "SENDER_2"}
    sender_1 = next(c for c in body if c["sender_id"] == "SENDER_1")
    assert sender_1["message_count"] == 2
    assert sender_1["last_message_preview"] == CANNED_REPLY
    assert sender_1["bot_disabled"] is False


def test_conversation_detail_not_found(client):
    resp = client.get("/admin/conversations/UNKNOWN", headers=AUTH)
    assert resp.status_code == 404


def test_conversation_detail_returns_messages(client):
    _seed("SENDER_1", ("user", "hi"), ("assistant", CANNED_REPLY))

    resp = client.get("/admin/conversations/SENDER_1", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["message_count"] == 2
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert [m["content"] for m in body["messages"]] == ["hi", CANNED_REPLY]


def test_reset_conversation_history(client):
    _seed("SENDER_1", ("user", "hi"), ("assistant", CANNED_REPLY))

    resp = client.delete("/admin/conversations/SENDER_1/history", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    assert client.get("/admin/conversations/SENDER_1", headers=AUTH).status_code == 404
