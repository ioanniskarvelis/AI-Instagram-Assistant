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


# ── Traces ───────────────────────────────────────────────────────────────


def _seed_trace(**overrides):
    import asyncio

    from app.trace import HistoryTurn, RetrievalHit, TraceRecord, save

    defaults = dict(
        sender_id="SENDER_1",
        incoming_text="Πόσο κοστίζει;",
        history_window=[HistoryTurn(role="user", text="Πόσο κοστίζει;")],
        intent="price",
        intent_latency_ms=10.0,
        retrieval_hits=[RetrievalHit(question="q", reply="r", score=0.9)],
        retrieval_latency_ms=5.0,
        system_prompt="SYSTEM PROMPT",
        reply="Η καλλιτέχνις θα σου απαντήσει.",
        reply_source="generated",
        llm_latency_ms=100.0,
        total_latency_ms=120.0,
    )
    defaults.update(overrides)
    asyncio.run(save(TraceRecord(**defaults)))


def test_traces_route_requires_auth(client):
    _seed_trace()
    assert client.get("/admin/traces").status_code == 401


def test_list_traces_returns_summaries(client):
    _seed_trace()
    resp = client.get("/admin/traces", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["sender_id"] == "SENDER_1"
    assert body[0]["intent"] == "price"
    assert body[0]["reply_source"] == "generated"
    assert "history_window" not in body[0]  # summary, not detail


def test_list_traces_filters_by_sender_id(client):
    _seed_trace(sender_id="SENDER_1")
    _seed_trace(sender_id="SENDER_2")

    resp = client.get("/admin/traces", params={"sender_id": "SENDER_2"}, headers=AUTH)
    body = resp.json()
    assert len(body) == 1
    assert body[0]["sender_id"] == "SENDER_2"


def test_trace_detail_returns_full_record(client):
    _seed_trace()
    trace_id = client.get("/admin/traces", headers=AUTH).json()[0]["id"]

    resp = client.get(f"/admin/traces/{trace_id}", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["system_prompt"] == "SYSTEM PROMPT"
    assert body["history_window"] == [{"role": "user", "text": "Πόσο κοστίζει;"}]
    assert body["retrieval_hits"] == [{"question": "q", "reply": "r", "score": 0.9}]
    assert body["intent_latency_ms"] == 10.0
    assert body["llm_latency_ms"] == 100.0


def test_trace_detail_not_found(client):
    resp = client.get("/admin/traces/999", headers=AUTH)
    assert resp.status_code == 404


def test_traces_ui_page_serves_without_auth(client):
    resp = client.get("/admin-ui/traces")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
