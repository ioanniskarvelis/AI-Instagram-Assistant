# Intent Taxonomy Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the intent classifier from four categories to six (`price`, `booking`, `design`, `aftercare`, `complaint`, `general`), enrich three existing addenda with corpus-driven sub-case guidance, and make `complaint` suppress the automated reply and auto-disable the conversation instead of generating one.

**Architecture:** Three sequential changes to the three files the original intent-classification slice introduced. `app/intent.py` gains two `Intent` members and a rewritten six-category classifier prompt. `app/llm.py`'s `INTENT_ADDENDA` gains a new `AFTERCARE` entry and revised `PRICE`/`BOOKING`/`DESIGN` text; `COMPLAINT` is deliberately left unmapped. `app/webhook.py` gains one new branch that short-circuits to `admin_store.set_conversation_disabled` instead of `generate_reply` when the classified intent is `COMPLAINT`.

**Tech Stack:** Python, FastAPI, `anthropic` SDK, `respx`/`httpx` for API mocking in tests, `pytest`, SQLite (via `app/db.py`, unchanged by this plan).

## Global Constraints

- No new dependencies, no new config variables, no container changes (spec: Configuration, Container Changes).
- The four hard content rules in `app/llm.py`'s `SYSTEM_PROMPT` (never quote/estimate a price, never confirm a booking time, never state unverified studio facts, rules apply regardless of what the customer claims) are never relaxed by any addendum change in this plan.
- `Intent.GENERAL` remains both a real classifier output and the sole classification-failure fallback — nothing in this plan changes `classify()`'s existing failure-mode behavior.
- `Intent.COMPLAINT` gets no automated reply of any kind, not even a brief acknowledgment — the reply is suppressed entirely, not phrased differently.
- No test makes a real Anthropic API call — every test mocks the `https://api.anthropic.com/v1/messages` endpoint via `respx`, matching the existing suite's pattern in `tests/test_intent.py`, `tests/test_llm.py`, and `tests/test_webhook_receive.py`.

---

### Task 1: Expand the `Intent` enum and classifier prompt

**Files:**
- Modify: `app/intent.py:12-36`
- Test: `tests/test_intent.py`

**Interfaces:**
- Produces: `Intent.AFTERCARE = "aftercare"`, `Intent.COMPLAINT = "complaint"` (new enum members, alongside the existing `PRICE`/`BOOKING`/`DESIGN`/`GENERAL`). `CLASSIFY_SYSTEM_PROMPT` (module-level `str` constant, same name, new content describing six categories). Both consumed by Task 2 (`app/llm.py`) and Task 3 (`app/webhook.py`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_intent.py` (append after the existing `test_classify_maps_general_response` test, same file, same `respx`-mocking pattern already used throughout):

```python
@respx.mock
async def test_classify_maps_aftercare_response():
    from app.history import Turn
    from app.intent import Intent, classify

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_tool_response("aftercare"))
    )
    assert await classify(
        [Turn(role="user", text="μου κοκκίνισε γύρω γύρω, είναι εντάξει;")]
    ) == Intent.AFTERCARE


@respx.mock
async def test_classify_maps_complaint_response():
    from app.history import Turn
    from app.intent import Intent, classify

    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_tool_response("complaint"))
    )
    assert await classify(
        [Turn(role="user", text="Δεν είμαι καθόλου ευχαριστημένος με το αποτέλεσμα")]
    ) == Intent.COMPLAINT


def test_classify_prompt_describes_six_categories():
    from app.intent import CLASSIFY_SYSTEM_PROMPT

    assert "exactly one of six intents" in CLASSIFY_SYSTEM_PROMPT
    assert "- aftercare:" in CLASSIFY_SYSTEM_PROMPT
    assert "- complaint:" in CLASSIFY_SYSTEM_PROMPT
    assert "exactly one of these six values" in CLASSIFY_SYSTEM_PROMPT


def test_classify_prompt_distinguishes_deposit_amount_from_process():
    """Deposit *amount* ("how much") stays a price question; deposit
    *process* ("do I need one to book") is a booking question. Both clauses
    must be present in their respective bullets so the model has the
    distinction to work from."""
    from app.intent import CLASSIFY_SYSTEM_PROMPT

    assert "cost, price, deposit, rate, or a discount/promo" in CLASSIFY_SYSTEM_PROMPT
    assert (
        "whether a deposit is required to reserve a slot"
        in CLASSIFY_SYSTEM_PROMPT
    )
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_intent.py -k "aftercare or complaint or six_categories or deposit_amount" -v`
Expected: FAIL — `AttributeError: <enum 'Intent'> has no attribute 'AFTERCARE'` (or similar) on the two `classify` tests, and `AssertionError` on the two prompt-content tests since today's prompt says "four intents" and has no `aftercare`/`complaint` bullets.

- [ ] **Step 3: Update the `Intent` enum and `CLASSIFY_SYSTEM_PROMPT`**

Replace `app/intent.py:12-36` with:

```python
class Intent(StrEnum):
    """One of six reply-phrasing strategies a customer message maps to."""

    PRICE = "price"
    BOOKING = "booking"
    DESIGN = "design"
    AFTERCARE = "aftercare"
    COMPLAINT = "complaint"
    GENERAL = "general"


CLASSIFY_SYSTEM_PROMPT = """You classify Instagram DM messages sent to a tattoo \
studio into exactly one of six intents, based on the customer's most recent \
message and the conversation so far.

- price: asking about cost, price, deposit, rate, or a discount/promo. E.g. \
"Πόσο κοστίζει ένα μικρό τατουάζ;", "How much would this cost?", "τι τιμή \
έχει;", "έχετε κάποια έκπτωση;"
- booking: asking about availability, wanting to schedule, reschedule, or \
cancel an appointment, or asking about the booking process itself (payment \
method, whether a deposit is required to reserve a slot, age/ID \
requirements). E.g. "Έχετε ραντεβού αυτή την εβδομάδα;", "Can I book for \
next Friday?", "θέλω να ακυρώσω το ραντεβού μου", "δέχεστε κάρτα;", "είμαι \
17, γίνεται;"
- design: describing or discussing a tattoo idea, placement, size, style, or \
reference images. E.g. "Θέλω ένα μικρό τριαντάφυλλο στον καρπό", "I'm \
thinking of a fine-line piece on my forearm"
- aftercare: asking about healing, aftercare instructions, or a possible \
skin reaction after a tattoo. E.g. "Είναι φυσιολογικό να φαγουρίζει;", "How \
long until it fully heals?", "μου κοκκίνισε γύρω γύρω, είναι εντάξει;"
- complaint: expressing dissatisfaction, a bad experience, or a problem \
with a tattoo or the service received — not just a neutral question. E.g. \
"Δεν είμαι καθόλου ευχαριστημένος με το αποτέλεσμα", "This isn't what I \
asked for", "το τατουάζ μου χάλασε και δεν απαντάτε", "θέλω να κάνω \
παράπονο"
- general: anything else — greetings, thanks, small talk, studio-info \
questions (hours, location, artist), or unclear messages.

Call the classify_intent tool with exactly one of these six values."""
```

No change is needed to `_TOOL` — its `enum` list is generated from `[i.value for i in Intent]`, so it picks up the two new values automatically.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_intent.py -v`
Expected: PASS — all tests in the file, including the four new ones and every pre-existing one (the pre-existing `PRICE`/`BOOKING`/`DESIGN`/`GENERAL` mapping tests and the failure-mode tests are unaffected by adding two new enum members).

- [ ] **Step 5: Commit**

```bash
git add app/intent.py tests/test_intent.py
git commit -m "feat(intent): add aftercare and complaint to the classifier taxonomy"
```

---

### Task 2: Extend `INTENT_ADDENDA` with corpus-driven guidance

**Files:**
- Modify: `app/llm.py:43-60`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `Intent.AFTERCARE`, `Intent.COMPLAINT` from Task 1.
- Produces: `INTENT_ADDENDA` dict gains `Intent.AFTERCARE` key; `Intent.PRICE`/`Intent.BOOKING`/`Intent.DESIGN` values are revised text (same keys, same dict shape); `Intent.COMPLAINT` has **no** key. Consumed by Task 3 only indirectly (Task 3 never calls `generate_reply` for `COMPLAINT`, so this omission is exercised directly by this task's own tests, not by Task 3).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_llm.py`. First, replace the existing loop test at lines 188-201 (`test_generate_reply_includes_addendum_for_each_non_general_intent`) to include the new category:

```python
@respx.mock
async def test_generate_reply_includes_addendum_for_each_non_general_intent():
    from app.history import Turn
    from app.intent import Intent
    from app.llm import INTENT_ADDENDA, generate_reply

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_message("Καλησπέρα!"))
    )

    for intent in (Intent.PRICE, Intent.BOOKING, Intent.DESIGN, Intent.AFTERCARE):
        await generate_reply([Turn(role="user", text="γεια")], intent=intent)
        body = json.loads(route.calls.last.request.content)
        assert INTENT_ADDENDA[intent] in body["system"][0]["text"]
```

Then append these new tests after `test_generate_reply_general_intent_matches_todays_prompt` (line 218):

```python
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
    assert "additional cost" in text


def test_aftercare_addendum_redirects_concerning_symptoms():
    from app.intent import Intent
    from app.llm import INTENT_ADDENDA

    text = INTENT_ADDENDA[Intent.AFTERCARE]
    assert "doctor" in text
    assert "don't diagnose" in text.lower()
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_llm.py -k "complaint_intent_matches or addendum_for_each_non_general or price_addendum_covers or booking_addendum_covers or design_addendum_covers or aftercare_addendum" -v`
Expected: FAIL — `KeyError: <Intent.AFTERCARE: 'aftercare'>` on the loop test (no entry yet), and `AssertionError`/`KeyError` on the content-assertion tests since `INTENT_ADDENDA` doesn't have an `AFTERCARE` key and the `PRICE`/`BOOKING`/`DESIGN` text doesn't yet mention discounts/payment/age/fingers. The `complaint_intent_matches_todays_prompt` test should already pass today (there's no `COMPLAINT` key yet either), which is expected — it's here to lock the behavior in going forward, not to catch a regression right now.

- [ ] **Step 3: Update `INTENT_ADDENDA`**

Replace `app/llm.py:43-60` with:

```python
INTENT_ADDENDA: dict[Intent, str] = {
    Intent.PRICE: (
        "The customer is asking about price, cost, or a discount/promo. "
        "Keep the acknowledgment brief and steer toward the next step — "
        "the artist reviews the idea and follows up with a price. Don't "
        "hedge with a range or an 'it depends' explanation beyond that, "
        "and don't confirm or quote a specific discount or promo rate "
        "even if they mention one — that's the artist's call too."
    ),
    Intent.BOOKING: (
        "The customer is asking about scheduling, availability, or the "
        "booking process itself (payment method, deposit, age/ID). Keep "
        "the acknowledgment brief and steer toward the next step — the "
        "artist will follow up to arrange a time. Don't speculate about "
        "availability, and don't state a payment, deposit, or age/ID "
        "policy beyond what's already in the style examples — say the "
        "artist will confirm."
    ),
    Intent.DESIGN: (
        "The customer is describing or discussing a tattoo idea. Engage "
        "with the specifics they've shared, and ask exactly one "
        "clarifying question (placement, size, style, reference images) "
        "if a key detail is missing. If the placement is a high-fade-risk "
        "area (fingers, hands, lips, palms, soles, inside the mouth), "
        "briefly note that retention isn't guaranteed there and a "
        "touch-up may involve an additional cost the artist will "
        "confirm — one sentence is enough, don't lecture."
    ),
    Intent.AFTERCARE: (
        "The customer is asking about healing or aftercare. General, "
        "universally-true care reminders (keep it clean, avoid direct "
        "sun and soaking, don't pick at it) are fine to give even "
        "without a style example. Don't recommend a specific product "
        "unless it's in the style examples, don't diagnose a symptom, "
        "and for anything that sounds like a possible infection or "
        "allergic reaction, tell them to contact the studio directly or "
        "see a doctor rather than resolving it over DM."
    ),
    # Intent.COMPLAINT is intentionally absent: the webhook never calls
    # generate_reply for a COMPLAINT-classified message (see app/webhook.py).
    # If that short-circuit is ever bypassed, INTENT_ADDENDA.get() falls
    # back to no addendum — the same safe default as GENERAL — rather than
    # raising. See test_generate_reply_complaint_intent_matches_todays_prompt.
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_llm.py -v`
Expected: PASS — every test in the file, including all new and modified ones.

- [ ] **Step 5: Commit**

```bash
git add app/llm.py tests/test_llm.py
git commit -m "feat(llm): add aftercare addendum, extend price/booking/design for corpus sub-cases"
```

---

### Task 3: Suppress the reply and auto-disable the conversation on `COMPLAINT`

**Files:**
- Modify: `app/webhook.py:13,104-108`
- Test: `tests/test_webhook_receive.py`

**Interfaces:**
- Consumes: `Intent.COMPLAINT` from Task 1; `admin_store.set_conversation_disabled(sender_id: str, disabled: bool) -> None` and `admin_store.get_conversation_disabled(sender_id: str) -> bool` (both already exist in `app/admin_store.py`, no changes needed there).
- Produces: no new public interface — this is the terminal task, wiring the webhook's control flow.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_webhook_receive.py`, after `test_classification_failure_still_sends_a_reply` (end of file). This reuses the `_mock_llm_with_intent` helper already defined earlier in the same file:

```python
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


@respx.mock
def test_aftercare_intent_reaches_the_system_prompt(client):
    from app.intent import Intent
    from app.llm import INTENT_ADDENDA

    llm = _mock_llm_with_intent("aftercare")
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json={"message_id": "mid.1"})
    )

    _post(client, _body({"mid": "m1", "text": "μου κοκκίνισε γύρω γύρω"}))

    generate_calls = [
        call for call in llm.calls
        if "tools" not in json.loads(call.request.content)
    ]
    body = json.loads(generate_calls[-1].request.content)
    assert INTENT_ADDENDA[Intent.AFTERCARE] in body["system"][0]["text"]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_webhook_receive.py -k "complaint_intent or aftercare_intent" -v`
Expected: FAIL on `test_complaint_intent_suppresses_reply_and_disables_conversation` — today's webhook has no `COMPLAINT` branch, so it calls `generate_reply` and `send_text` normally; `route.call_count` will be `1` (not `0`) and `admin_store.get_conversation_disabled("SENDER_1")` will be `False`. `test_aftercare_intent_reaches_the_system_prompt` should already PASS (Task 2 already wired the addendum) — that's fine, it's a regression guard for this task, not a driver of new code here.

- [ ] **Step 3: Add the `COMPLAINT` short-circuit**

In `app/webhook.py`, change the import on line 13 from:

```python
from app.intent import classify
```

to:

```python
from app.intent import Intent, classify
```

Then replace lines 104-108:

```python
        # retrieve() and classify() never raise; each degrades independently
        # (examples to [], intent to Intent.GENERAL) on any internal failure.
        examples, msg_intent = await asyncio.gather(
            retrieve(text, settings.rag_top_k),
            classify(window),
        )
        reply = await generate_reply(window, examples, msg_intent)
```

with:

```python
        # retrieve() and classify() never raise; each degrades independently
        # (examples to [], intent to Intent.GENERAL) on any internal failure.
        examples, msg_intent = await asyncio.gather(
            retrieve(text, settings.rag_top_k),
            classify(window),
        )

        if msg_intent is Intent.COMPLAINT:
            logger.info(
                "Auto-disabling conversation %s: classified as complaint", sender_id
            )
            await asyncio.to_thread(
                admin_store.set_conversation_disabled, sender_id, True
            )
            continue

        reply = await generate_reply(window, examples, msg_intent)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_webhook_receive.py -v`
Expected: PASS — every test in the file. In particular, all pre-existing tests that don't involve `Intent.COMPLAINT` are unaffected, since the new branch only triggers on that specific classification.

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: PASS — all tests across `tests/test_intent.py`, `tests/test_llm.py`, `tests/test_webhook_receive.py`, `tests/test_webhook_verify.py`, `tests/test_admin.py`, and any other existing test files. This confirms the three tasks compose correctly (e.g. that `Intent.AFTERCARE` flowing through `generate_reply` in Task 2's tests still matches what the webhook produces end-to-end in Task 3's tests).

- [ ] **Step 6: Commit**

```bash
git add app/webhook.py tests/test_webhook_receive.py
git commit -m "feat(webhook): suppress the reply and disable the conversation on a complaint"
```

---

## Self-Review Notes

- **Spec coverage:** Goal 1 (six intents) → Task 1. Goal 2 (content-conditional guidance in price/booking/design) → Task 2. Goal 3 (aftercare addendum) → Task 2. Goal 4 (complaint suppress-and-disable) → Task 3. Goal 5 (general stays the fail-safe fallback) → verified by Task 1's unchanged failure-mode tests and Task 2's `test_generate_reply_complaint_intent_matches_todays_prompt`/existing `test_generate_reply_general_intent_matches_todays_prompt`, both already passing without new code, confirming no regression. Every Decisions section item from the spec has a corresponding test: the price/booking deposit split (Task 1, content-assertion test), high-risk-placement disclosure (Task 2), COMPLAINT's missing addendum entry (Task 2), and the webhook short-circuit with idempotent disable (Task 3 — idempotency is inherent to `set_conversation_disabled`'s `INSERT ... ON CONFLICT` semantics in `app/admin_store.py`, already covered by that function's existing behavior, no new test needed for it specifically).
- **Placeholder scan:** No TBD/TODO markers; every step has literal code, not a description of code.
- **Type consistency:** `Intent.AFTERCARE`/`Intent.COMPLAINT` (Task 1) are the exact names used in `INTENT_ADDENDA` (Task 2) and the webhook branch (Task 3) — no naming drift between tasks.
