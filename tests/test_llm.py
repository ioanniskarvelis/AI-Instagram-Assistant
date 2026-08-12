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
    assert await generate_reply(turns) == "Καλησπέρα!"

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

    assert await generate_reply([Turn(role="user", text="γεια")]) is None


@respx.mock
async def test_generate_reply_returns_none_on_transport_error():
    from app.history import Turn
    from app.llm import generate_reply

    respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("boom"))

    assert await generate_reply([Turn(role="user", text="γεια")]) is None


@respx.mock
async def test_generate_reply_returns_none_on_refusal():
    from app.history import Turn
    from app.llm import generate_reply

    refusal = _message("", stop_reason="refusal")
    refusal["content"] = []
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=refusal))

    assert await generate_reply([Turn(role="user", text="γεια")]) is None


@respx.mock
async def test_generate_reply_returns_none_on_truncation():
    from app.history import Turn
    from app.llm import generate_reply

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200, json=_message("Η τιμή ξεκινάει από", stop_reason="max_tokens")
        )
    )

    assert await generate_reply([Turn(role="user", text="γεια")]) is None


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

    assert reply == "Καλησπέρα!"
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
