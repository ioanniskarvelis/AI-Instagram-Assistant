import asyncio
import json
import time

import httpx
import respx
from fastapi.testclient import TestClient

from tests.conftest import CANNED_REPLY, sign


def _run(coro):
    """Run an async app function (e.g. app.quotes' asyncio.to_thread-backed
    helpers) from these synchronous TestClient-based tests."""
    return asyncio.run(coro)


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


def _wait_idle(timeout: float = 15.0) -> None:
    """Block until every sender's debounce task has fired and cleaned itself
    up. The app's TestClient runs the ASGI app on its own event loop in a
    background thread, so from here a plain polling wait is the simplest way
    to synchronize with that detached background task.

    15s of headroom (not the usual sub-second debounce/test budget) because
    a couple of tests deliberately provoke a 500 from the Anthropic API,
    and the SDK's own retry-with-backoff runs to completion before
    `generate_reply`/`classify` return control to us.
    """
    from app.webhook import _pending

    deadline = time.monotonic() + timeout
    while _pending:
        if time.monotonic() > deadline:
            raise TimeoutError(f"debounce tasks still pending: {list(_pending)}")
        time.sleep(0.01)


def _post(client, body: bytes, signature: str | None = None):
    headers = {"Content-Type": "application/json"}
    headers["X-Hub-Signature-256"] = sign(body) if signature is None else signature
    response = client.post("/webhook", content=body, headers=headers)
    _wait_idle()
    return response


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


@respx.mock
def test_style_examples_reach_the_system_prompt(client, monkeypatch, tmp_path):
    from app.config import get_settings
    from app.rag import _load_index

    index_path = tmp_path / "rag_index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "question": "Γεια σας",
                    "reply": "Καλησπέρα, πώς μπορούμε να βοηθήσουμε;",
                    "embedding": [1.0, 0.0],
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("RAG_INDEX_PATH", str(index_path))
    get_settings.cache_clear()
    _load_index.cache_clear()

    respx.post("https://openrouter.ai/api/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "embedding": [1.0, 0.0], "index": 0}],
                "model": "voyageai/voyage-4",
                "usage": {"total_tokens": 1},
            },
        )
    )
    llm = _mock_llm()
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    _post(client, _body({"mid": "m1", "text": "Γεία σας"}))

    body = json.loads(llm.calls.last.request.content)
    assert "Καλησπέρα, πώς μπορούμε να βοηθήσουμε;" in body["system"][0]["text"]


@respx.mock
def test_no_style_examples_when_openrouter_key_unset(client):
    """Today's behavior, unchanged: no OPENROUTER_API_KEY means no RAG block."""
    llm = _mock_llm()
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    _post(client, _body({"mid": "m1", "text": "Γεία σας"}))

    body = json.loads(llm.calls.last.request.content)
    assert "STYLE REFERENCE" not in body["system"][0]["text"]


def _tool_response(intent_value: str) -> dict:
    return {
        "id": "msg_03",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5-20251001",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "classify_intent",
                "input": {"intent": intent_value},
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _mock_llm_with_intent(intent_value: str, reply_text: str = GENERATED):
    """Mocks both concurrent Anthropic calls (classify + generate) on the
    same endpoint, routing by request shape: a classification call carries
    "tools"/forced tool_choice, a generation call does not."""

    def responder(request):
        body = json.loads(request.content)
        if "tools" in body:
            return httpx.Response(200, json=_tool_response(intent_value))
        return httpx.Response(200, json=_anthropic_reply(reply_text))

    return respx.post(ANTHROPIC_ENDPOINT).mock(side_effect=responder)


@respx.mock
def test_intent_addendum_reaches_the_system_prompt(client):
    from app.intent import Intent
    from app.llm import INTENT_ADDENDA

    llm = _mock_llm_with_intent("price")
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    _post(client, _body({"mid": "m1", "text": "Πόσο κοστίζει ένα τατουάζ;"}))

    generate_calls = [
        call for call in llm.calls
        if "tools" not in json.loads(call.request.content)
    ]
    body = json.loads(generate_calls[-1].request.content)
    assert INTENT_ADDENDA[Intent.PRICE] in body["system"][0]["text"]


@respx.mock
def test_general_intent_adds_no_addendum(client):
    from app.llm import SYSTEM_PROMPT

    llm = _mock_llm_with_intent("general")
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    _post(client, _body({"mid": "m1", "text": "Γεια σας"}))

    generate_calls = [
        call for call in llm.calls
        if "tools" not in json.loads(call.request.content)
    ]
    body = json.loads(generate_calls[-1].request.content)
    assert body["system"][0]["text"] == SYSTEM_PROMPT


@respx.mock
def test_classification_failure_still_sends_a_reply(client):
    """A broken classifier degrades to Intent.GENERAL — same never-block
    posture as a broken RAG index. The reply still goes out."""
    from app.llm import SYSTEM_PROMPT

    def responder(request):
        body = json.loads(request.content)
        if "tools" in body:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json=_anthropic_reply())

    llm = respx.post(ANTHROPIC_ENDPOINT).mock(side_effect=responder)
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    response = _post(client, _body({"mid": "m1", "text": "Γεια σας"}))

    assert response.status_code == 200
    sent = json.loads(route.calls.last.request.content)
    assert sent["message"]["text"] == GENERATED

    generate_calls = [
        call for call in llm.calls
        if "tools" not in json.loads(call.request.content)
    ]
    body = json.loads(generate_calls[-1].request.content)
    assert body["system"][0]["text"] == SYSTEM_PROMPT


@respx.mock
def test_complaint_intent_suppresses_reply_and_disables_conversation(client):
    from app import admin_store

    llm = _mock_llm_with_intent("complaint")
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    response = _post(
        client,
        _body({"mid": "m1", "text": "Δεν είμαι καθόλου ευχαριστημένος με το αποτέλεσμα"}),
    )

    assert response.status_code == 200
    assert route.call_count == 0  # no Instagram send
    assert admin_store.get_conversation_disabled("SENDER_1") is True
    assert _stored("SENDER_1") == [
        ("user", "Δεν είμαι καθόλου ευχαριστημένος με το αποτέλεσμα")
    ]

    generate_calls = [
        call for call in llm.calls
        if "tools" not in json.loads(call.request.content)
    ]
    assert generate_calls == []  # generate_reply's API call never fired


def _traces(sender_id: str) -> list[dict]:
    from app.trace import list_traces

    return [
        {"intent": t.intent, "reply_source": t.reply_source, "reply": t.reply}
        for t in list_traces(limit=10, offset=0, sender_id=sender_id)
    ]


@respx.mock
def test_generated_reply_produces_a_trace_row(client):
    _mock_llm_with_intent("price")
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    _post(client, _body({"mid": "m1", "text": "Πόσο κοστίζει;"}))

    assert _traces("SENDER_1") == [
        {"intent": "price", "reply_source": "generated", "reply": GENERATED}
    ]


@respx.mock
def test_canned_fallback_produces_a_trace_row(client):
    respx.post(ANTHROPIC_ENDPOINT).mock(
        return_value=httpx.Response(
            400,
            json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}},
        )
    )
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    _post(client, _body({"mid": "m1", "text": "Γεια σας"}))

    traces = _traces("SENDER_1")
    assert len(traces) == 1
    assert traces[0]["reply_source"] == "canned"
    assert traces[0]["reply"] == CANNED_REPLY


@respx.mock
def test_complaint_produces_a_suppressed_trace_row_with_no_reply(client):
    _mock_llm_with_intent("complaint")
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    _post(client, _body({"mid": "m1", "text": "Δεν είμαι καθόλου ευχαριστημένος"}))

    assert _traces("SENDER_1") == [
        {"intent": "complaint", "reply_source": "suppressed", "reply": None}
    ]


@respx.mock
def test_burst_of_messages_coalesces_into_one_reply(env, monkeypatch):
    """Two messages sent within the debounce window produce exactly one
    reply, generated from their concatenation, once the window elapses."""
    from app.config import get_settings

    monkeypatch.setenv("REPLY_DEBOUNCE_SECONDS", "0.2")
    get_settings.cache_clear()

    from app.main import create_app
    from app import webhook

    seen_text: list[str] = []

    async def _fake_retrieve_scored(text, k):
        seen_text.append(text)
        return []

    monkeypatch.setattr(webhook, "retrieve_scored", _fake_retrieve_scored)

    _mock_llm()
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    with TestClient(create_app()) as client:
        headers = {"Content-Type": "application/json"}
        first = _body({"mid": "m1", "text": "τατουάζ"})
        second = _body({"mid": "m2", "text": "στο χέρι"})

        client.post(
            "/webhook", content=first,
            headers={**headers, "X-Hub-Signature-256": sign(first)},
        )
        client.post(
            "/webhook", content=second,
            headers={**headers, "X-Hub-Signature-256": sign(second)},
        )
        _wait_idle()

    assert route.call_count == 1
    assert seen_text == ["τατουάζ\nστο χέρι"]
    assert _stored("SENDER_1") == [
        ("user", "τατουάζ"),
        ("user", "στο χέρι"),
        ("assistant", GENERATED),
    ]


@respx.mock
def test_burst_with_an_image_coalesces_into_one_reply(env, monkeypatch):
    """An image dropped mid-burst is neither lost nor split into its own
    reply: it becomes a placeholder turn that joins the same debounced
    burst as the surrounding text messages."""
    from app.config import get_settings

    monkeypatch.setenv("REPLY_DEBOUNCE_SECONDS", "0.2")
    get_settings.cache_clear()

    from app.main import create_app
    from app import webhook

    seen_text: list[str] = []

    async def _fake_retrieve_scored(text, k):
        seen_text.append(text)
        return []

    monkeypatch.setattr(webhook, "retrieve_scored", _fake_retrieve_scored)

    _mock_llm()
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    with TestClient(create_app()) as client:
        headers = {"Content-Type": "application/json"}
        first = _body({"mid": "m1", "text": "τατουάζ"})
        image = _body({"mid": "m2", "attachments": [{"type": "image"}]})
        third = _body({"mid": "m3", "text": "στο χέρι"})

        for body in (first, image, third):
            client.post(
                "/webhook", content=body,
                headers={**headers, "X-Hub-Signature-256": sign(body)},
            )
        _wait_idle()

    assert route.call_count == 1
    assert seen_text == ["τατουάζ\n[the customer sent an image]\nστο χέρι"]
    assert _stored("SENDER_1") == [
        ("user", "τατουάζ"),
        ("user", "[the customer sent an image]"),
        ("user", "στο χέρι"),
        ("assistant", GENERATED),
    ]


@respx.mock
def test_aftercare_intent_reaches_the_system_prompt(client):
    from app.intent import Intent
    from app.llm import INTENT_ADDENDA

    llm = _mock_llm_with_intent("aftercare")
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    _post(client, _body({"mid": "m1", "text": "μου κοκκίνισε γύρω γύρω"}))

    assert route.call_count == 1  # reply was actually sent, unlike complaint
    generate_calls = [
        call for call in llm.calls
        if "tools" not in json.loads(call.request.content)
    ]
    body = json.loads(generate_calls[-1].request.content)
    assert INTENT_ADDENDA[Intent.AFTERCARE] in body["system"][0]["text"]


TELEGRAM_BASE = "https://api.telegram.org/botTEST_TOKEN"


def _configured_telegram_client(monkeypatch):
    from app.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TEST_TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
    get_settings.cache_clear()

    return TestClient(create_app())


def _mock_llm_with_intent_and_quote(intent_value: str, reply_text: str, summary: str):
    """Like _mock_llm_with_intent, but the generate call (the one carrying
    the request_quote tool, once Telegram is configured) returns both reply
    text and a request_quote tool_use block instead of plain text."""

    def responder(request):
        body = json.loads(request.content)
        tools = body.get("tools")
        if tools and tools[0]["name"] == "classify_intent":
            return httpx.Response(200, json=_tool_response(intent_value))
        return httpx.Response(
            200,
            json={
                "id": "msg_04",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [
                    {"type": "text", "text": reply_text},
                    {
                        "type": "tool_use",
                        "id": "toolu_02",
                        "name": "request_quote",
                        "input": {"summary": summary},
                    },
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    return respx.post(ANTHROPIC_ENDPOINT).mock(side_effect=responder)


@respx.mock
def test_gathered_details_trigger_a_telegram_quote_request(env, monkeypatch):
    _mock_llm_with_intent_and_quote(
        "design", "Ένα λεπτό, το ελέγχουμε με την ομάδα! 🐼", "small rose, forearm, 5cm"
    )
    ig_route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )
    telegram_message_route = respx.post(f"{TELEGRAM_BASE}/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 777}})
    )

    with _configured_telegram_client(monkeypatch) as client:
        response = _post(
            client, _body({"mid": "m1", "text": "μικρό τριαντάφυλλο στον καρπό, 5cm"})
        )

        assert response.status_code == 200
        # The customer gets the model's own accompanying text, not a canned line.
        sent = json.loads(ig_route.calls.last.request.content)
        assert sent["message"]["text"] == "Ένα λεπτό, το ελέγχουμε με την ομάδα! 🐼"

        assert telegram_message_route.called
        telegram_body = json.loads(telegram_message_route.calls.last.request.content)
        assert telegram_body["chat_id"] == "-100123"
        assert "small rose, forearm, 5cm" in telegram_body["text"]
        assert "SENDER_1" in telegram_body["text"]

        from app import quotes
        assert _run(quotes.resolve_quote_request(777)) == "SENDER_1"


@respx.mock
def test_quote_request_includes_reference_images_sent_earlier(env, monkeypatch):
    """The image and the follow-up text arrive within the debounce window
    (like the earlier burst-coalescing tests) so they settle into a single
    reply/quote-request, rather than the image alone triggering its own."""
    from app.config import get_settings

    monkeypatch.setenv("REPLY_DEBOUNCE_SECONDS", "0.2")
    get_settings.cache_clear()

    _mock_llm_with_intent_and_quote("design", "Ένα λεπτό! 🐼", "rose, forearm, 5cm")
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )
    photo_route = respx.post(f"{TELEGRAM_BASE}/sendPhoto").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})
    )
    message_route = respx.post(f"{TELEGRAM_BASE}/sendMessage").mock(
        return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 778}})
    )

    with _configured_telegram_client(monkeypatch) as client:
        headers = {"Content-Type": "application/json"}
        image_body = _body({"mid": "m1", "attachments": [
            {"type": "image", "payload": {"url": "https://cdn/ref.jpg"}}
        ]})
        text_body = _body({"mid": "m2", "text": "rose, forearm, 5cm"})

        for body in (image_body, text_body):
            client.post(
                "/webhook", content=body,
                headers={**headers, "X-Hub-Signature-256": sign(body)},
            )
        _wait_idle()

    assert message_route.call_count == 1  # one quote request, not two
    photo_body = json.loads(photo_route.calls.last.request.content)
    assert photo_body["photo"] == "https://cdn/ref.jpg"


@respx.mock
def test_quote_request_failure_still_delivers_the_customer_reply(env, monkeypatch):
    _mock_llm_with_intent_and_quote(
        "design", "Ένα λεπτό, το ελέγχουμε με την ομάδα! 🐼", "rose, forearm, 5cm"
    )
    ig_route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )
    respx.post(f"{TELEGRAM_BASE}/sendMessage").mock(
        return_value=httpx.Response(500, json={"ok": False})
    )

    with _configured_telegram_client(monkeypatch) as client:
        response = _post(client, _body({"mid": "m1", "text": "rose, forearm, 5cm"}))

    assert response.status_code == 200
    sent = json.loads(ig_route.calls.last.request.content)
    assert sent["message"]["text"] == "Ένα λεπτό, το ελέγχουμε με την ομάδα! 🐼"


def test_image_attachments_are_recorded_for_later_quote_use(client):
    body = _body(
        {"mid": "m1", "attachments": [{"type": "image", "payload": {"url": "https://cdn/ref.jpg"}}]}
    )
    headers = {"Content-Type": "application/json"}
    response = client.post(
        "/webhook", content=body, headers={**headers, "X-Hub-Signature-256": sign(body)}
    )
    _wait_idle()

    assert response.status_code == 200
    from app import quotes
    assert _run(quotes.image_urls_for("SENDER_1")) == ["https://cdn/ref.jpg"]
