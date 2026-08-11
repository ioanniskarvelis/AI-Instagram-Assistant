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
