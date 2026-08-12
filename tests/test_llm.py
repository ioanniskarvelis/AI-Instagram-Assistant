import json

import httpx
import respx

ENDPOINT = "https://api.anthropic.com/v1/messages"


def _message(text: str, stop_reason: str = "end_turn") -> dict:
    """A minimal but schema-valid Messages API response."""
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


@respx.mock
async def test_generate_reply_returns_text_and_sends_the_window():
    from app.history import Turn
    from app.llm import SYSTEM_PROMPT, generate_reply

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Καλησπέρα!"))
    )

    turns = [
        Turn(role="user", text="γεια"),
        Turn(role="assistant", text="γεια σου"),
        Turn(role="user", text="πόσο κάνει;"),
    ]
    assert (await generate_reply(turns)).text == "Καλησπέρα!"

    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "claude-sonnet-5"
    assert body["max_tokens"] == 2000
    assert body["output_config"] == {"effort": "low"}
    assert body["system"][0]["text"] == SYSTEM_PROMPT
    assert body["messages"] == [
        {"role": "user", "content": "γεια"},
        {"role": "assistant", "content": "γεια σου"},
        {"role": "user", "content": "πόσο κάνει;"},
    ]
    # Thinking must never be disabled — effort is the cost lever.
    assert body.get("thinking") is None


@respx.mock
async def test_generate_reply_returns_none_on_api_error():
    from app.history import Turn
    from app.llm import generate_reply

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            400, json={"type": "error", "error": {"type": "invalid_request_error",
                                                  "message": "bad"}}
        )
    )

    assert (await generate_reply([Turn(role="user", text="γεια")])).text is None


@respx.mock
async def test_generate_reply_returns_none_on_transport_error():
    from app.history import Turn
    from app.llm import generate_reply

    respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("boom"))

    assert (await generate_reply([Turn(role="user", text="γεια")])).text is None


@respx.mock
async def test_generate_reply_returns_none_on_refusal():
    from app.history import Turn
    from app.llm import generate_reply

    refusal = _message("", stop_reason="refusal")
    refusal["content"] = []
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=refusal))

    assert (await generate_reply([Turn(role="user", text="γεια")])).text is None


@respx.mock
async def test_generate_reply_returns_none_on_truncation():
    from app.history import Turn
    from app.llm import generate_reply

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200, json=_message("Η τιμή ξεκινάει από", stop_reason="max_tokens")
        )
    )

    assert (await generate_reply([Turn(role="user", text="γεια")])).text is None


@respx.mock
async def test_generate_reply_includes_style_examples_after_rules():
    from app.history import Turn
    from app.llm import SYSTEM_PROMPT, generate_reply
    from app.rag import Example

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Καλησπέρα!"))
    )

    examples = [
        Example(question="Πόσο κοστίζει;", reply="Στείλε φωτο και σου λέμε [price]"),
    ]
    await generate_reply([Turn(role="user", text="γεια")], examples)

    body = json.loads(route.calls.last.request.content)
    system_text = body["system"][0]["text"]
    assert system_text.startswith(SYSTEM_PROMPT)
    style_index = system_text.index("STYLE REFERENCE")
    assert style_index > len(SYSTEM_PROMPT)
    assert "Πόσο κοστίζει;" in system_text
    assert "Στείλε φωτο και σου λέμε [price]" in system_text


@respx.mock
async def test_generate_reply_truncates_long_examples_in_style_block():
    from app.history import Turn
    from app.llm import MAX_EXAMPLE_FIELD_CHARS, generate_reply
    from app.rag import Example

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Καλησπέρα!"))
    )

    long_reply = "x" * (MAX_EXAMPLE_FIELD_CHARS + 50)
    short_reply = "a short reply"
    examples = [
        Example(question="short question", reply=long_reply),
        Example(question="another short question", reply=short_reply),
    ]
    await generate_reply([Turn(role="user", text="γεια")], examples)

    body = json.loads(route.calls.last.request.content)
    system_text = body["system"][0]["text"]

    assert long_reply not in system_text
    assert ("x" * (MAX_EXAMPLE_FIELD_CHARS - 1) + "…") in system_text
    assert short_reply in system_text


@respx.mock
async def test_generate_reply_without_examples_matches_todays_prompt():
    from app.history import Turn
    from app.llm import SYSTEM_PROMPT, generate_reply

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Καλησπέρα!"))
    )

    await generate_reply([Turn(role="user", text="γεια")])

    body = json.loads(route.calls.last.request.content)
    assert body["system"][0]["text"] == SYSTEM_PROMPT


@respx.mock
async def test_generate_reply_includes_intent_addendum_after_rules():
    from app.history import Turn
    from app.intent import Intent
    from app.llm import INTENT_ADDENDA, SYSTEM_PROMPT, generate_reply

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Καλησπέρα!"))
    )

    await generate_reply([Turn(role="user", text="πόσο κοστίζει;")], intent=Intent.PRICE)

    body = json.loads(route.calls.last.request.content)
    system_text = body["system"][0]["text"]
    assert system_text.startswith(SYSTEM_PROMPT)
    addendum_index = system_text.index(INTENT_ADDENDA[Intent.PRICE])
    assert addendum_index > len(SYSTEM_PROMPT)


@respx.mock
async def test_generate_reply_includes_addendum_for_each_non_general_intent():
    from app.history import Turn
    from app.intent import Intent
    from app.llm import INTENT_ADDENDA, generate_reply

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Καλησπέρα!"))
    )

    for intent in (Intent.PRICE, Intent.BOOKING, Intent.DESIGN, Intent.AFTERCARE):
        await generate_reply([Turn(role="user", text="γεία")], intent=intent)
        body = json.loads(route.calls.last.request.content)
        assert INTENT_ADDENDA[intent] in body["system"][0]["text"]


@respx.mock
async def test_generate_reply_general_intent_matches_todays_prompt():
    from app.history import Turn
    from app.intent import Intent
    from app.llm import SYSTEM_PROMPT, generate_reply

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Καλησπέρα!"))
    )

    await generate_reply([Turn(role="user", text="γεια")], intent=Intent.GENERAL)

    body = json.loads(route.calls.last.request.content)
    assert body["system"][0]["text"] == SYSTEM_PROMPT


@respx.mock
async def test_generate_reply_complaint_intent_matches_todays_prompt():
    """COMPLAINT has no INTENT_ADDENDA entry — generate_reply must degrade to
    the base prompt exactly like GENERAL, via the same INTENT_ADDENDA.get()
    path already covered by test_generate_reply_unmapped_intent_degrades_to_no_addendum.
    This is the defense-in-depth guarantee: even if the webhook's complaint
    short-circuit is ever bypassed by a bug, the fallback reply carries no
    complaint-specific phrasing."""
    from app.history import Turn
    from app.intent import Intent
    from app.llm import SYSTEM_PROMPT, generate_reply

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Καλησπέρα!"))
    )

    await generate_reply([Turn(role="user", text="γεια")], intent=Intent.COMPLAINT)

    body = json.loads(route.calls.last.request.content)
    assert body["system"][0]["text"] == SYSTEM_PROMPT


def test_price_addendum_covers_discounts():
    from app.intent import Intent
    from app.llm import INTENT_ADDENDA

    assert "discount" in INTENT_ADDENDA[Intent.PRICE]


def test_booking_addendum_covers_payment_and_age():
    from app.intent import Intent
    from app.llm import INTENT_ADDENDA

    text = INTENT_ADDENDA[Intent.BOOKING]
    assert "payment" in text
    assert "age" in text


def test_design_addendum_covers_high_risk_placement():
    from app.intent import Intent
    from app.llm import INTENT_ADDENDA

    text = INTENT_ADDENDA[Intent.DESIGN]
    assert "fingers" in text
    assert "touch-up" in text
    assert "artist will confirm" in text


def test_aftercare_addendum_redirects_concerning_symptoms():
    from app.intent import Intent
    from app.llm import INTENT_ADDENDA

    text = INTENT_ADDENDA[Intent.AFTERCARE]
    assert "doctor" in text
    assert "don't diagnose" in text.lower()


@respx.mock
async def test_generate_reply_unmapped_intent_degrades_to_no_addendum():
    """INTENT_ADDENDA has no entry for a value like this (Intent is closed to
    exactly PRICE/BOOKING/DESIGN/AFTERCARE/COMPLAINT/GENERAL today, so
    nothing produces this at runtime yet) — generate_reply must degrade to
    the base prompt via
    INTENT_ADDENDA.get(), never raise a KeyError, if the enum ever grows
    without a matching addendum."""
    from app.history import Turn
    from app.llm import INTENT_ADDENDA, SYSTEM_PROMPT, generate_reply

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Καλησπέρα!"))
    )

    unmapped = "future_intent_with_no_addendum"
    assert unmapped not in INTENT_ADDENDA

    reply = await generate_reply([Turn(role="user", text="γεια")], intent=unmapped)

    assert reply.text == "Καλησπέρα!"
    body = json.loads(route.calls.last.request.content)
    assert body["system"][0]["text"] == SYSTEM_PROMPT


@respx.mock
async def test_generate_reply_intent_addendum_precedes_style_block():
    from app.history import Turn
    from app.intent import Intent
    from app.llm import INTENT_ADDENDA, generate_reply
    from app.rag import Example

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Καλησπέρα!"))
    )

    examples = [
        Example(question="Πόσο κοστίζει;", reply="Στείλε φωτο και σου λέμε [price]"),
    ]
    await generate_reply(
        [Turn(role="user", text="πόσο κοστίζει;")], examples, Intent.PRICE
    )

    body = json.loads(route.calls.last.request.content)
    system_text = body["system"][0]["text"]
    addendum_index = system_text.index(INTENT_ADDENDA[Intent.PRICE])
    style_index = system_text.index("STYLE REFERENCE")
    assert addendum_index < style_index


def _tool_use(name: str, input_: dict, stop_reason: str = "tool_use") -> dict:
    return {
        "id": "msg_02",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [
            {"type": "tool_use", "id": "toolu_01", "name": name, "input": input_}
        ],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _text_and_tool_use(text: str, name: str, input_: dict) -> dict:
    return {
        "id": "msg_03",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [
            {"type": "text", "text": text},
            {"type": "tool_use", "id": "toolu_01", "name": name, "input": input_},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


@respx.mock
async def test_generate_reply_omits_quote_tool_when_telegram_not_configured():
    from app.history import Turn
    from app.llm import generate_reply

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Καλησπέρα!"))
    )

    await generate_reply([Turn(role="user", text="γεια")])

    body = json.loads(route.calls.last.request.content)
    assert "tools" not in body


@respx.mock
async def test_generate_reply_offers_quote_tool_when_telegram_configured(env, monkeypatch):
    from app.config import get_settings
    from app.history import Turn
    from app.llm import QUOTE_TOOL_NAME, generate_reply

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TEST_TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    get_settings.cache_clear()

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Καλησπέρα!"))
    )

    await generate_reply([Turn(role="user", text="γεια")])

    body = json.loads(route.calls.last.request.content)
    assert body["tools"][0]["name"] == QUOTE_TOOL_NAME
    assert body["tool_choice"] == {"type": "auto"}


@respx.mock
async def test_generate_reply_returns_text_and_quote_summary_together(env, monkeypatch):
    from app.config import get_settings
    from app.history import Turn
    from app.llm import generate_reply

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TEST_TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    get_settings.cache_clear()

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json=_text_and_tool_use(
                "Ένα λεπτό, το ελέγχουμε με την ομάδα! 🐼",
                "request_quote",
                {"summary": "small rose, forearm, 5cm, reference image provided"},
            ),
        )
    )

    result = await generate_reply([Turn(role="user", text="γεια")])

    assert result.text == "Ένα λεπτό, το ελέγχουμε με την ομάδα! 🐼"
    assert result.quote_summary == "small rose, forearm, 5cm, reference image provided"


@respx.mock
async def test_generate_reply_tool_only_response_has_no_text(env, monkeypatch):
    from app.config import get_settings
    from app.history import Turn
    from app.llm import generate_reply

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TEST_TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100123")
    get_settings.cache_clear()

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200, json=_tool_use("request_quote", {"summary": "small rose, forearm"})
        )
    )

    result = await generate_reply([Turn(role="user", text="γεια")])

    assert result.text is None
    assert result.quote_summary == "small rose, forearm"


@respx.mock
async def test_generate_quote_announcement_appends_a_new_instruction_turn():
    from app.history import Turn
    from app.llm import generate_quote_announcement

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Η τιμή είναι 150€! 🐼"))
    )

    turns = [
        Turn(role="user", text="θέλω ένα τριαντάφυλλο"),
        Turn(role="assistant", text="ωραία, στείλε φωτο"),
    ]
    reply = await generate_quote_announcement(turns, "150")

    assert reply == "Η τιμή είναι 150€! 🐼"
    body = json.loads(route.calls.last.request.content)
    assert len(body["messages"]) == 3
    assert body["messages"][-1]["role"] == "user"
    assert "150" in body["messages"][-1]["content"]
    assert "[INTERNAL — artist quote]" in body["messages"][-1]["content"]


@respx.mock
async def test_generate_quote_announcement_merges_into_trailing_user_turn():
    """Anthropic rejects two consecutive same-role messages, so when the
    window's last turn is already the customer's (no assistant reply sent
    yet), the instruction must merge into it rather than append a second
    user turn."""
    from app.history import Turn
    from app.llm import generate_quote_announcement

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Η τιμή είναι 150€! 🐼"))
    )

    turns = [Turn(role="user", text="θέλω ένα τριαντάφυλλο")]
    await generate_quote_announcement(turns, "150")

    body = json.loads(route.calls.last.request.content)
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    assert "θέλω ένα τριαντάφυλλο" in body["messages"][0]["content"]
    assert "150" in body["messages"][0]["content"]


@respx.mock
async def test_generate_quote_announcement_returns_none_on_failure():
    from app.history import Turn
    from app.llm import generate_quote_announcement

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            400,
            json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}},
        )
    )

    reply = await generate_quote_announcement(
        [Turn(role="user", text="γεια")], "150"
    )

    assert reply is None
