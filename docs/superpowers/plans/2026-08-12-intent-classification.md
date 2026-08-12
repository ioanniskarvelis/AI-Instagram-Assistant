# Intent Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify each inbound customer DM into one of four intents
(`price`, `booking`, `design`, `general`) and layer an intent-specific
phrasing-strategy addendum into the reply-generation system prompt, without
touching the existing hard content rules.

**Architecture:** One new runtime module, `app/intent.py`, makes a forced
`tool_use` Anthropic call against a cheap/fast model to classify the
customer's message. `app/webhook.py` runs that call concurrently with the
existing RAG retrieval call (`asyncio.gather`), then passes both results into
`app/llm.py`'s `generate_reply`, which inserts the matching addendum between
the hard rules and the RAG style block.

**Tech Stack:** FastAPI, `anthropic` SDK (same client pattern as
`app/llm.py`'s `generate_reply`), `respx`/`httpx` for mocking the Anthropic
Messages API in tests — no new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-12-intent-classification-design.md`

## Global Constraints

- Classification must never block or degrade a reply. Any failure (API
  error, refusal, missing/malformed tool_use block) makes `classify()`
  return `Intent.GENERAL` — never raises. This matches the existing
  never-raise contract of `app/rag.py`'s `retrieve()` and `app/llm.py`'s
  `generate_reply()`.
- `Intent.GENERAL` produces **no** system-prompt addendum — identical to
  today's prompt. This is both a real classifier output and the fallback
  value, so "classified as general" and "classification failed" are
  indistinguishable to the customer.
- The existing hard rules in `SYSTEM_PROMPT` (no prices, no booking
  confirmations, no unverified facts) are never modified. Intent addenda are
  additive phrasing-strategy guidance, inserted after the hard rules.
- System prompt composition order when both are present: hard rules → intent
  addendum → RAG style block.
- Classification runs concurrently with RAG retrieval via `asyncio.gather`,
  not sequentially — it must not add latency beyond whichever of the two
  calls is slower.
- No new "enable/disable" config flag. Classification reuses the
  already-required `ANTHROPIC_API_KEY`; there's no unconfigured state to
  degrade from.
- No test makes a real Anthropic API call — mock via `respx` against
  `https://api.anthropic.com/v1/messages`, the same pattern already used in
  `tests/test_llm.py` and `tests/test_webhook_receive.py`.

---

### Task 1: Intent classification configuration settings

**Files:**
- Modify: `app/config.py`
- Modify: `tests/conftest.py`
- Modify: `.env.example`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.intent_model: str` (default
  `"claude-haiku-4-5-20251001"`), `Settings.intent_max_tokens: int` (default
  `50`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_intent_settings_apply_documented_defaults():
    from app.config import get_settings

    settings = get_settings()
    assert settings.intent_model == "claude-haiku-4-5-20251001"
    assert settings.intent_max_tokens == 50
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_config.py::test_intent_settings_apply_documented_defaults -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'intent_model'`

- [ ] **Step 3: Add the settings**

In `app/config.py`, add after the `rag_index_path` field (inside the
`Settings` class):

```python
    # Intent classification (price/booking/design/general), run before every
    # reply to steer its phrasing strategy. Deliberately a separate, cheaper/
    # faster model from ANTHROPIC_MODEL — this call happens on every message.
    intent_model: str = "claude-haiku-4-5-20251001"
    intent_max_tokens: int = 50
```

- [ ] **Step 4: Pin the new settings in the test environment**

In `tests/conftest.py`, add to the `env` fixture, after the existing
`RAG_INDEX_PATH` line:

```python
    monkeypatch.setenv("INTENT_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("INTENT_MAX_TOKENS", "50")
```

- [ ] **Step 5: Document the new variables**

In `.env.example`, add after the existing RAG block (after `RAG_TOP_K=3`):

```
# Optional — model and token cap for per-message intent classification (see
# docs/superpowers/specs/2026-08-12-intent-classification-design.md). Cheap
# and fast on purpose: this runs on every message, separate from
# ANTHROPIC_MODEL which handles the higher-quality reply generation call.
INTENT_MODEL=claude-haiku-4-5-20251001
INTENT_MAX_TOKENS=50
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: PASS (all tests in the file, including the new one)

- [ ] **Step 7: Commit**

```bash
git add app/config.py tests/conftest.py tests/test_config.py .env.example
git commit -m "feat(config): add intent classification settings (model, max tokens)"
```

---

### Task 2: `app/intent.py` — intent classification

**Files:**
- Create: `app/intent.py`
- Test: `tests/test_intent.py`

**Interfaces:**
- Consumes: `Settings.intent_model`, `Settings.intent_max_tokens` from Task 1;
  `app.history.Turn` (`role: str`, `text: str`).
- Produces: `Intent` (`StrEnum`: `PRICE = "price"`, `BOOKING = "booking"`,
  `DESIGN = "design"`, `GENERAL = "general"`); `CLASSIFY_SYSTEM_PROMPT: str`;
  `async classify(turns: list[Turn]) -> Intent` — never raises, returns
  `Intent.GENERAL` on any failure.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_intent.py`:

```python
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
    assert await classify([Turn(role="user", text="γεια")]) == Intent.GENERAL
```

This will fail to import (`app.intent` does not exist yet).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_intent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.intent'`

- [ ] **Step 3: Write `app/intent.py`**

```python
import logging
from enum import StrEnum

import anthropic

from app.config import get_settings
from app.history import Turn

logger = logging.getLogger(__name__)


class Intent(StrEnum):
    """One of four reply-phrasing strategies a customer message maps to."""

    PRICE = "price"
    BOOKING = "booking"
    DESIGN = "design"
    GENERAL = "general"


CLASSIFY_SYSTEM_PROMPT = """You classify Instagram DM messages sent to a tattoo \
studio into exactly one of four intents, based on the customer's most recent \
message and the conversation so far.

- price: asking about cost, price, deposit, or rate. E.g. "Πόσο κοστίζει ένα \
μικρό τατουάζ;", "How much would this cost?", "τι τιμή έχει;"
- booking: asking about availability, wanting to schedule, reschedule, or \
cancel an appointment. E.g. "Έχετε ραντεβού αυτή την εβδομάδα;", "Can I book \
for next Friday?", "θέλω να ακυρώσω το ραντεβού μου"
- design: describing or discussing a tattoo idea, placement, size, style, or \
reference images. E.g. "Θέλω ένα μικρό τριαντάφυλλο στον καρπό", "I'm thinking \
of a fine-line piece on my forearm"
- general: anything else — greetings, thanks, small talk, studio-info \
questions (hours, location, artist), or unclear messages.

Call the classify_intent tool with exactly one of these four values."""

_TOOL_NAME = "classify_intent"
_TOOL = {
    "name": _TOOL_NAME,
    "description": "Record the classified intent of the customer's message.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [i.value for i in Intent],
            }
        },
        "required": ["intent"],
    },
}

_client = anthropic.AsyncAnthropic(api_key=get_settings().anthropic_api_key)


async def classify(turns: list[Turn]) -> Intent:
    """Classify the customer's intent from the recent conversation window.

    Returns Intent.GENERAL on any failure — API error, refusal, a response
    with no tool_use block, or an unrecognized intent value. Never raises:
    a classification problem must never block a reply from being generated,
    the same degrade-silently posture as app.rag.retrieve returning [].
    """
    try:
        settings = get_settings()
        response = await _client.messages.create(
            model=settings.intent_model,
            max_tokens=settings.intent_max_tokens,
            system=[
                {
                    "type": "text",
                    "text": CLASSIFY_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=[{"role": turn.role, "content": turn.text} for turn in turns],
        )
    except anthropic.APIError:
        logger.exception("Anthropic call failed during intent classification")
        return Intent.GENERAL
    except Exception:
        logger.exception("Unexpected error classifying intent")
        return Intent.GENERAL

    if response.stop_reason == "refusal":
        logger.warning("Model declined to classify (stop_reason=refusal)")
        return Intent.GENERAL

    for block in response.content:
        if block.type == "tool_use" and block.name == _TOOL_NAME:
            value = block.input.get("intent")
            try:
                return Intent(value)
            except ValueError:
                logger.warning("Unrecognized intent value from classifier: %r", value)
                return Intent.GENERAL

    logger.warning(
        "No tool_use block in classification response (stop_reason=%s)",
        response.stop_reason,
    )
    return Intent.GENERAL
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_intent.py -v`
Expected: PASS (all 10 tests)

- [ ] **Step 5: Commit**

```bash
git add app/intent.py tests/test_intent.py
git commit -m "feat(intent): classify customer messages into price/booking/design/general"
```

---

### Task 3: Layer the intent addendum into the system prompt

**Files:**
- Modify: `app/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `Intent` from Task 2.
- Produces: `generate_reply(turns, examples=None, intent=None)` — `intent`
  parameter added, defaulting to `None` (treated identically to
  `Intent.GENERAL`); `INTENT_ADDENDA: dict[Intent, str]` (no entry for
  `Intent.GENERAL` — that key is intentionally absent, not empty-string).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_llm.py`:

```python
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

    for intent in (Intent.PRICE, Intent.BOOKING, Intent.DESIGN):
        await generate_reply([Turn(role="user", text="γεια")], intent=intent)
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_llm.py -v`
Expected: FAIL — `generate_reply() got an unexpected keyword argument 'intent'`

- [ ] **Step 3: Modify `app/llm.py`**

Add the import (alongside the existing `app.rag` import):

```python
from app.intent import Intent
```

Add after the `STYLE_BLOCK_FOOTER` definition (around `app/llm.py:37-40`):

```python
INTENT_ADDENDA: dict[Intent, str] = {
    Intent.PRICE: (
        "The customer is asking about price or cost. Keep the acknowledgment "
        "brief and steer toward the next step — the artist reviews the idea "
        "and follows up with a price. Don't hedge with a range or an "
        "'it depends' explanation beyond that."
    ),
    Intent.BOOKING: (
        "The customer is asking about scheduling or availability. Keep the "
        "acknowledgment brief and steer toward the next step — the artist "
        "will follow up to arrange a time. Don't speculate about availability."
    ),
    Intent.DESIGN: (
        "The customer is describing or discussing a tattoo idea. Engage with "
        "the specifics they've shared, and ask exactly one clarifying question "
        "(placement, size, style, reference images) if a key detail is missing."
    ),
}
```

Replace the `generate_reply` signature and its system-prompt-building lines
(`app/llm.py:63-74`):

```python
async def generate_reply(
    turns: list[Turn],
    examples: list[Example] | None = None,
    intent: Intent | None = None,
) -> str | None:
    """Generate a reply from the conversation window.

    Returns None on any failure. Never raises: the webhook must return 200 to
    Meta whether or not generation worked, and the caller falls back to the
    canned reply.
    """
    parts = [SYSTEM_PROMPT]
    if intent is not None and intent != Intent.GENERAL:
        parts.append(INTENT_ADDENDA[intent])
    if examples:
        parts.append(_render_style_block(examples))
    system_text = "\n\n".join(parts)
```

The rest of the function body (the `try`/`except` call, the `stop_reason`
checks, the return) is unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_llm.py -v`
Expected: PASS (all tests in the file — existing and new)

- [ ] **Step 5: Commit**

```bash
git add app/llm.py tests/test_llm.py
git commit -m "feat(llm): layer an intent-specific addendum into the system prompt"
```

---

### Task 4: Wire classification into the webhook flow

**Files:**
- Modify: `app/webhook.py`
- Test: `tests/test_webhook_receive.py`

**Interfaces:**
- Consumes: `classify` from Task 2, `generate_reply(turns, examples, intent)`
  from Task 3.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_webhook_receive.py`:

```python
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
```

Note: all the existing tests in this file that call `_mock_llm()` (not
`_mock_llm_with_intent()`) continue to work unmodified — their single static
mock response has no `tools`/`tool_use` block, so `classify()` hits the
"no tool_use block in the response" branch and degrades to `Intent.GENERAL`,
which is exactly today's prompt. No existing assertions change.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_webhook_receive.py -v`
Expected: The three new tests FAIL (`ImportError`/`ModuleNotFoundError` for
`app.intent`, or the addendum text is simply absent from the system prompt
since nothing calls `classify` yet); all pre-existing tests in the file still
PASS.

- [ ] **Step 3: Modify `app/webhook.py`**

Add the import (alongside the existing `app.rag` import):

```python
from app.intent import classify
```

Replace the retrieval-and-generation lines (`app/webhook.py:101-103`):

```python
        # retrieve() and classify() never raise; each degrades independently
        # (examples to [], intent to Intent.GENERAL) on any internal failure.
        examples, msg_intent = await asyncio.gather(
            retrieve(text, settings.rag_top_k),
            classify(window),
        )
        reply = await generate_reply(window, examples, msg_intent)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_webhook_receive.py -v`
Expected: PASS (all tests in the file — existing and new)

- [ ] **Step 5: Run the full test suite**

Run: `.venv/Scripts/python -m pytest -v`
Expected: PASS (every test in `tests/`, no regressions)

- [ ] **Step 6: Commit**

```bash
git add app/webhook.py tests/test_webhook_receive.py
git commit -m "feat(webhook): classify intent concurrently with RAG retrieval before generating a reply"
```
