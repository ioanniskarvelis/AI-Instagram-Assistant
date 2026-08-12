import json

import httpx
import respx

ENDPOINT = "https://api.anthropic.com/v1/messages"


def _tool_response(intent_value: str) -> dict:
    """A minimal but schema-valid Messages API response with a forced
    tool_use block, as a classification call would return."""
    return {
        "id": "msg_01",
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


def _text_response(stop_reason: str = "end_turn") -> dict:
    """Simulates the model ignoring tool_choice and returning plain text —
    the defensive case classify() must still degrade gracefully from."""
    return {
        "id": "msg_02",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5-20251001",
        "content": [] if stop_reason == "refusal" else [{"type": "text", "text": "ok"}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


@respx.mock
async def test_classify_sends_the_conversation_window_and_forces_the_tool():
    from app.history import Turn
    from app.intent import CLASSIFY_SYSTEM_PROMPT, Intent, classify

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_tool_response("price"))
    )

    turns = [
        Turn(role="user", text="γεια"),
        Turn(role="assistant", text="γεια σου"),
        Turn(role="user", text="πόσο κάνει ένα μικρό τατουάζ;"),
    ]
    assert await classify(turns) == Intent.PRICE

    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "claude-haiku-4-5-20251001"
    assert body["max_tokens"] == 50
    assert body["system"][0]["text"] == CLASSIFY_SYSTEM_PROMPT
    assert body["tools"][0]["name"] == "classify_intent"
    assert body["tool_choice"] == {"type": "tool", "name": "classify_intent"}
    assert body["messages"] == [
        {"role": "user", "content": "γεια"},
        {"role": "assistant", "content": "γεια σου"},
        {"role": "user", "content": "πόσο κάνει ένα μικρό τατουάζ;"},
    ]


@respx.mock
async def test_classify_maps_booking_response():
    from app.history import Turn
    from app.intent import Intent, classify

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_tool_response("booking"))
    )
    assert await classify([Turn(role="user", text="έχετε ραντεβού αύριο;")]) == Intent.BOOKING


@respx.mock
async def test_classify_maps_design_response():
    from app.history import Turn
    from app.intent import Intent, classify

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_tool_response("design"))
    )
    assert await classify([Turn(role="user", text="θέλω ένα τριαντάφυλλο")]) == Intent.DESIGN


@respx.mock
async def test_classify_maps_general_response():
    from app.history import Turn
    from app.intent import Intent, classify

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_tool_response("general"))
    )
    assert await classify([Turn(role="user", text="γεια σας")]) == Intent.GENERAL


@respx.mock
async def test_classify_uses_prior_turns_for_a_context_dependent_message():
    """A bare "ναι" only makes sense with the preceding turn visible — this
    confirms classify() sends the full window, not just the last message, so
    a context-dependent reply like this has a chance of being classified
    correctly (here: continuing a booking thread)."""
    from app.history import Turn
    from app.intent import Intent, classify

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_tool_response("booking"))
    )

    turns = [
        Turn(role="user", text="έχετε ραντεβού την Παρασκευή;"),
        Turn(role="assistant", text="Ναι, τι ώρα σας βολεύει;"),
        Turn(role="user", text="ναι"),
    ]
    assert await classify(turns) == Intent.BOOKING

    body = json.loads(route.calls.last.request.content)
    assert body["messages"] == [
        {"role": "user", "content": "έχετε ραντεβού την Παρασκευή;"},
        {"role": "assistant", "content": "Ναι, τι ώρα σας βολεύει;"},
        {"role": "user", "content": "ναι"},
    ]


@respx.mock
async def test_classify_returns_general_on_api_error():
    from app.history import Turn
    from app.intent import Intent, classify

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            400,
            json={"type": "error", "error": {"type": "invalid_request_error",
                                              "message": "bad"}},
        )
    )
    assert await classify([Turn(role="user", text="γεια")]) == Intent.GENERAL


@respx.mock
async def test_classify_returns_general_on_transport_error():
    from app.history import Turn
    from app.intent import Intent, classify

    respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("boom"))
    assert await classify([Turn(role="user", text="γεια")]) == Intent.GENERAL


@respx.mock
async def test_classify_returns_general_on_refusal():
    from app.history import Turn
    from app.intent import Intent, classify

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_text_response(stop_reason="refusal"))
    )
    assert await classify([Turn(role="user", text="γεια")]) == Intent.GENERAL


@respx.mock
async def test_classify_returns_general_when_no_tool_use_block():
    from app.history import Turn
    from app.intent import Intent, classify

    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_text_response()))
    assert await classify([Turn(role="user", text="γεια")]) == Intent.GENERAL


@respx.mock
async def test_classify_returns_general_for_unrecognized_intent_value():
    from app.history import Turn
    from app.intent import Intent, classify

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_tool_response("not_a_real_intent"))
    )
    assert await classify([Turn(role="user", text="γεία")]) == Intent.GENERAL
