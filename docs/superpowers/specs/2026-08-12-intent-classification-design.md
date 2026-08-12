# Instagram Assistant — Intent Classification

**Date:** 2026-08-12
**Status:** Approved, ready for planning
**Builds on:** `docs/superpowers/specs/2026-08-12-rag-style-retrieval-design.md`

## Context

Every inbound DM is currently handled by one fixed `SYSTEM_PROMPT` in `app/llm.py`,
regardless of what the customer is actually asking about. The hard rules (never
quote a price, never confirm a booking time, never state unverified studio facts)
are correctly universal — they don't change per message. But the *reply strategy*
around those rules is currently generic, whether the customer is asking "how much
does this cost," describing a tattoo idea in detail, or just saying "hi."

This slice adds a lightweight intent-classification step before reply generation,
so the assistant can steer its phrasing strategy differently depending on what
kind of message it's replying to, without touching the hard content rules.

## Goals

1. Classify each inbound customer message into one of four intents: `price`,
   `booking`, `design`, `general`.
2. Layer intent-specific phrasing guidance into the system prompt, in addition to
   (never instead of) the existing hard rules.
3. Add no meaningful latency to the reply path — classification runs concurrently
   with the existing RAG retrieval call, not sequentially before it.
4. Degrade to today's behavior (no addendum, i.e. `general`) on any classification
   failure — a classifier outage must never block or degrade the reply itself.

## Non-Goals

- Routing any intent to a human instead of auto-replying. Every intent still gets
  an automated reply; only the phrasing strategy changes. (A future slice could
  add human handoff for specific intents, but that's out of scope here.)
- Changing the hard content rules (no prices, no booking confirmations, no
  unverified facts) for any intent. Those apply identically regardless of intent.
- Skipping RAG retrieval or history persistence for any intent. Both happen for
  every message exactly as they do today.
- Multi-label or confidence-scored classification. Exactly one intent per
  message, forced by the tool schema.
- Admin-visible intent tagging/analytics. The label is consumed internally by
  `generate_reply` and not persisted or surfaced anywhere.

## Decisions

### Four intents, one of which is a no-op

`price`, `booking`, and `design` each get a short addendum block appended to the
system prompt. `general` gets no addendum at all — it is both a real classifier
output (greetings, small talk, studio-info questions, anything that doesn't fit
the other three) and the fallback value on any classification failure. Making
"classified as general" and "classification failed" produce an identical prompt
means a classifier outage is genuinely invisible to reply quality, not just
non-blocking.

### Separate cheap/fast model, forced tool_use, enum-constrained

Classification runs as its own Anthropic call against a distinct, cheaper/faster
model (`intent_model` in config) rather than reusing `anthropic_model` — this call
happens on every message and doing a full-quality generation-tier call just to
pick one of four labels would double the model cost per message for no quality
benefit.

The call forces `tool_choice` on a single tool whose `intent` parameter is
constrained to the four category strings. This guarantees a valid label back
without free-text parsing/validation — the same reliability property forced
tool_use gives any structured-extraction task.

### Classifier sees the conversation window, not just the latest message

Unlike `rag.retrieve` (which embeds only the incoming message text), `intent.classify`
receives the same recent-history window `generate_reply` already gets. Some
messages are only classifiable with context — a bare "ναι" ("yes") or "τι ώρα;"
("what time?") that continues an existing design or booking thread reads
differently in isolation than with the preceding turns visible.

### Concurrent with RAG retrieval, not sequential

`intent.classify` and `rag.retrieve` are independent network calls with no data
dependency on each other. The webhook runs them via `asyncio.gather` rather than
awaiting one after the other, so this slice adds effectively zero latency beyond
whichever of the two calls is slower — not the sum of both.

### Addendum inserted between hard rules and the RAG style block

System prompt composition order becomes: `SYSTEM_PROMPT` (hard rules) → intent
addendum (if any) → RAG style block (if any). Behavioral guidance (what strategy
to use) belongs before stylistic guidance (how past replies sounded), and hard
rules always come first regardless of what's layered after them.

### No feature flag / enable switch

RAG needed an empty-key escape hatch because it depends on an optional third-party
API key (`OPENROUTER_API_KEY`) that might not be configured. Intent classification
reuses the already-required `anthropic_api_key`, so there's no "unconfigured"
state to degrade from — it's on whenever the assistant runs at all. Failures are
handled per-call (fallback to `general`), not via a global toggle.

## Architecture

```
app/webhook.py
   │  asyncio.gather(
   │      rag.retrieve(text, k),        # existing
   │      intent.classify(window),      # new
   │  )
   ▼
app/llm.py generate_reply(turns, examples, intent)
   │  system prompt = SYSTEM_PROMPT
   │                + INTENT_ADDENDA[intent]   (if intent != GENERAL)
   │                + style block               (if examples)
   ▼
Anthropic reply generation (unchanged)
```

| Module | Responsibility | Depends on |
|---|---|---|
| `app/intent.py` (new) | Defines `Intent` enum; `classify(turns) -> Intent` via a forced-tool-use Anthropic call against `intent_model`. Never raises — returns `Intent.GENERAL` on any failure. | `config`, `history.Turn`, `anthropic` |
| `app/llm.py` (modified) | `generate_reply` gains an `intent` parameter; builds the intent addendum block when `intent` is not `GENERAL`, inserted before the RAG style block. | `intent` (only for the `Intent` type) |
| `app/webhook.py` (modified) | Calls `intent.classify` concurrently with `rag.retrieve` via `asyncio.gather`; passes both results to `generate_reply`. | `intent` |

## Interfaces

### `app/intent.py`

```python
class Intent(StrEnum):
    PRICE = "price"
    BOOKING = "booking"
    DESIGN = "design"
    GENERAL = "general"

async def classify(turns: list[Turn]) -> Intent:
    """Classify the customer's intent from the recent conversation window.

    Never raises: returns Intent.GENERAL on any failure (API error, refusal,
    missing/malformed tool_use block, or unexpected exception), the same
    degrade-silently posture as rag.retrieve returning [].
    """
```

- Builds a dedicated classification system prompt (distinct from `SYSTEM_PROMPT`)
  describing the four categories with brief examples per category (Greek +
  English, matching the studio's actual customer base).
- Calls `anthropic.AsyncAnthropic.messages.create` with:
  - `model=settings.intent_model`
  - `max_tokens=settings.intent_max_tokens`
  - a single tool (`classify_intent`) whose `intent` input parameter is a JSON
    Schema enum of the four values
  - `tool_choice` forcing that tool
  - `messages` built from `turns`, same shape as `generate_reply` uses
- Reads the forced tool_use block's `intent` argument off the response and
  returns the corresponding `Intent`. Any deviation from the happy path (API
  error, `stop_reason == "refusal"`, no tool_use block present, an argument
  value that isn't one of the four, or any other exception) is logged and
  returns `Intent.GENERAL`.

### `app/llm.py`

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

async def generate_reply(
    turns: list[Turn],
    examples: list[Example] | None = None,
    intent: Intent | None = None,
) -> str | None:
```

- When `intent` is present and not `Intent.GENERAL`, its addendum from
  `INTENT_ADDENDA` is inserted between `SYSTEM_PROMPT` and the RAG style block
  (each present or absent independently — same "no examples → no style block"
  behavior as today, now mirrored for intent).
- `intent=None` (e.g. a caller that hasn't adopted classification) behaves
  identically to `intent=Intent.GENERAL` — no addendum.

### `app/webhook.py`

Replaces the current:
```python
examples = await retrieve(text, settings.rag_top_k)
reply = await generate_reply(window, examples)
```
with:
```python
examples, msg_intent = await asyncio.gather(
    retrieve(text, settings.rag_top_k),
    classify(window),
)
reply = await generate_reply(window, examples, msg_intent)
```

## Configuration

Added to `app/config.py` / `.env`:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `intent_model` | no | `claude-haiku-4-5-20251001` | Model used for the classification call — cheap/fast, separate from `anthropic_model`. Confirm this is still the current Haiku model id during implementation; it's configuration, not a code change, if it's moved on. |
| `intent_max_tokens` | no | `50` | Caps the forced tool_use response |

Both fall under the existing required `anthropic_api_key` — no new credential.

## Error Handling

| Failure | Behavior |
|---|---|
| Anthropic API error/timeout on the classification call | Logged; `classify` returns `Intent.GENERAL` for that message only |
| `stop_reason == "refusal"` | Logged; `Intent.GENERAL` |
| No tool_use block in the response, or an unrecognized `intent` argument value | Logged; `Intent.GENERAL` |
| Any other unexpected exception | Logged; `Intent.GENERAL` |

No new failure path reaches the customer differently than today: a classification
failure produces the same prompt as an explicit `general` classification, which
in turn produces the same prompt as today's code before this slice existed. The
existing `generate_reply` failure/fallback-to-`CANNED_REPLY` behavior is
unchanged.

## Testing

- `app/intent.py`: forced-tool-use call returns the correct `Intent` for
  representative Greek and English fixtures per category (including a
  context-dependent case like a bare "ναι"/"yes" that only makes sense with the
  preceding turn); falls back to `Intent.GENERAL` on a mocked API error, on a
  refusal `stop_reason`, and on a response with no tool_use block.
- `app/llm.py`: system prompt contains the correct addendum text for each
  non-`GENERAL` intent, no addendum for `GENERAL` or `None`, and — when both an
  intent addendum and RAG examples are present — the addendum appears **after**
  the hard rules and **before** the style block.
- `app/webhook.py`: both `classify` and `retrieve` are invoked and awaited
  concurrently (not sequentially), and both results reach `generate_reply`.
- No test makes a real Anthropic API call — same mocking pattern already used
  for `generate_reply` and `rag.retrieve` in the existing suite.

## Container Changes

None. No new dependencies, no new files, no new bind mounts — this is pure
application code plus two config variables with defaults.
