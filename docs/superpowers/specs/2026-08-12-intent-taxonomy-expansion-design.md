# Instagram Assistant — Intent Taxonomy Expansion

**Date:** 2026-08-12
**Status:** Approved, ready for planning
**Builds on:** `docs/superpowers/specs/2026-08-12-intent-classification-design.md`

## Context

The first slice of intent classification shipped four intents — `price`,
`booking`, `design`, `general` — each guessed from first principles about what
a tattoo studio's DMs probably contain. With the RAG corpus now extracted
(`data/rag_corpus_approved.jsonl`, 5,128 real customer↔studio message pairs),
we can check that guess against the studio's actual DM history instead of
guessing further.

A keyword-frequency pass plus spot-reading of the corpus confirmed the
original four intents cover the bulk of message volume correctly (cancel and
reschedule already read naturally as `booking`; hours/location already read
naturally as `general`). It also surfaced recurring patterns with no current
guidance:

- **Aftercare/healing questions** (~91 occurrences) — a distinct,
  informational topic currently falling into `general` with zero specific
  guidance, which risks the model improvising medical-sounding advice.
- **A high-risk-placement disclaimer** — for finger/lip/hand tattoos, the
  human handler consistently appends a liability disclaimer (fading risk,
  informed consent, paid touch-up) that `design`'s addendum doesn't direct
  the model to replicate.
- **Deposit/payment-process questions**, distinct from "how much is the
  deposit" (already a `price` question) — "do you need a deposit to hold my
  slot," "do you take card," which are booking-process questions, not price
  figures.
- **Age/ID verification for minors** — legally sensitive, low-volume, and
  usually raised mid-conversation rather than as a standalone message.
- **Discount/promo requests** (~16 occurrences) — real (the studio runs
  promos) but not explicitly addressed by `price`'s existing addendum.
- **Complaints/negative feedback** — only one example in the corpus, but the
  human reply is a deliberately different move: de-escalate, offer a free
  fix, never get defensive. Same low-frequency/high-stakes asymmetry that
  justifies the SYSTEM_PROMPT's existing hard rules.

This slice expands the taxonomy and the addenda to close these gaps, and adds
one deliberate control-flow exception for complaints.

## Goals

1. Expand the classifier from four intents to six: add `aftercare` and
   `complaint`.
2. Extend `price`, `booking`, and `design`'s addenda with content-conditional
   guidance for the sub-cases found in the corpus (discounts; deposit/payment
   process and age/ID; high-risk-placement disclosure) — handled by the
   full-quality generation model reading message content, not by asking the
   cheap classifier to detect finer-grained sub-categories.
3. Add an `aftercare` addendum: general universally-true care reminders are
   fine to state unprompted, but no unverified product recommendations, no
   symptom diagnosis, and concerning symptoms get redirected to the studio or
   a doctor rather than resolved over DM.
4. For `complaint` specifically: suppress the automated reply entirely and
   auto-disable the conversation via the existing admin kill switch, so a
   human takes over that thread. This is a deliberate, scoped exception to
   the original slice's non-goal of "every intent still gets an automated
   reply" — complaints are the one case where an automated reply is the wrong
   move regardless of how well-phrased it is.
5. Preserve `general` as both a real classifier output and the
   classification-failure fallback, unchanged from the original design. This
   matters more now than before: a classifier failure must never resolve to
   `complaint` (which suppresses the reply) or any other consequential
   category — it must degrade to today's plain behavior, exactly as
   `rag.retrieve` degrades to `[]`.

## Non-Goals

- Distinguishing complaint-triggered conversation disables from manual admin
  disables. Both reuse the same `conversation_state.disabled` flag and look
  identical on the dashboard — no new column, no schema change. The studio
  opens the conversation and reads the messages to see why it's disabled,
  the same way they already would for a manual pause.
- Proactively notifying staff (push, email, SMS, etc.) when a conversation
  auto-disables. Staff discover it by checking the dashboard, exactly as
  they already do for manually paused conversations. This is an accepted
  limitation, not a deferred requirement — see Error Handling.
- Any reply at all — including a brief empathetic acknowledgment — for a
  `complaint`-classified message. The conversation goes silent immediately;
  the human writes the first response.
- Relaxing any of the four hard content rules in `SYSTEM_PROMPT` (never quote
  a price, never confirm a booking time/date, never state unverified studio
  facts, rules apply regardless of what the customer claims). The new
  addenda direct *where* those rules apply with concrete phrasing; none of
  them create an exception to them.
- A dedicated classifier category for age/ID verification, or for splitting
  deposit-amount vs. deposit-process into different categories beyond what's
  specified below. Also skipped: gift vouchers, cover-ups, laser-removal
  follow-ups — each under 40 corpus occurrences (several single digits),
  too rare to justify a dedicated category or addendum change.
- Routing any intent other than `complaint` to a human. Every other intent
  still gets an automated reply, exactly as the original design specified.

## Decisions

### Six categories: two new, three refined, one unchanged

`aftercare` and `complaint` become new classifier outputs — both are
genuinely distinct customer topics with genuinely distinct reply strategies,
not sub-cases of an existing category. Everything else found in the corpus
(discounts, deposit process, age/ID, high-risk placement) stays inside an
existing category's addendum as content-conditional guidance: the customer's
underlying topic is still price, booking, or design, and the full-quality
generation model already sees the entire message, not just a classifier
label, so it doesn't need a new category to act on a sub-case correctly.
Adding a classifier category per sub-case would only add misclassification
surface for no behavioral gain. `general` is untouched — same definition,
same no-op addendum, same fallback role.

### `price` keeps deposit *amount* and discounts; `booking` gains deposit and age/ID *process*

The classifier's `price` definition already listed "deposit" as an example
before this slice — a "how much is the deposit" question was, and remains,
`price`. This slice adds discount/promo questions to that same bucket, since
they're the same shape of question (a figure the model must not quote) and
belong to the same addendum. `booking`'s definition is extended to cover
*process* questions about the same topics — "do you need a deposit to hold my
slot," "do you take card," "I'm 17, is that OK" — which aren't asking for a
number, they're asking how the booking mechanics work. This split is new and
easy to get backwards, so it's called out explicitly for the test plan below.

### High-risk-placement disclosure lives in `design`'s addendum, not a new rule

Fingers, hands, lips, palms, soles, and inside-the-mouth placements carry a
real fading/retention risk the studio's human handler consistently discloses.
This is new information the model must volunteer, not just a restriction —
so unlike the deposit/age clauses (which redirect the model to defer, per the
existing hard rules), this one asks the model to proactively add one sentence
when the described placement matches. Phrased as "may involve an additional
cost the artist will confirm," never a number or estimate — consistent with,
not an exception to, the hard price rule.

### `aftercare` distinguishes universal advice from studio-specific claims

Generic care reminders (keep it clean, avoid direct sun and soaking, don't
pick at it) are true regardless of which studio said them, so the model may
state them without a style example backing it up — unlike a *studio-specific*
product recommendation, which falls under the existing hard rule against
unverified studio facts. Symptom questions (possible infection, allergic
reaction) are explicitly routed to "contact the studio or see a doctor," not
resolved conversationally — this is the one place in the addenda where
"don't try to be helpful, defer instead" is the safer instruction, mirroring
why the hard rules exist for price and booking.

### `complaint` changes control flow, not phrasing

Every other intent in this design — old and new — only changes *how* the
reply is phrased. `complaint` is the one deliberate exception: no reply is
generated at all, and the webhook auto-disables the conversation via
`admin_store.set_conversation_disabled`, the same function the studio's
dashboard already calls for a manual pause. This reuses an existing,
already-tested primitive rather than introducing a new one. Consequently,
`Intent.COMPLAINT` has **no entry** in `INTENT_ADDENDA` — `generate_reply` is
never expected to be called with it. If that expectation is ever violated by
a future bug, `INTENT_ADDENDA.get(intent)` already returns `None` for any
unmapped key (the existing `4b3db51` guard), so the safe failure mode is "a
plain reply, no addendum" — the same as `general` — not a crash.

### `general` remains the sole fail-safe fallback

`intent.classify` already returns `Intent.GENERAL` on every failure mode
(API error, refusal, missing/malformed tool_use block, unrecognized value).
That is unchanged by adding two more valid categories. This is what keeps
`complaint`'s new suppress-and-disable behavior safe under classifier
failure: a classification outage degrades to today's plain-reply behavior,
never to silence.

### Accepted risk: false-positive `complaint` classification goes silent

A message misclassified as `complaint` gets zero reply and the thread goes
silent until a human happens to check the dashboard — a real behavior change
from today, where every allowed message gets some reply. The mitigation is
entirely in the classify-prompt's example quality (requiring clear
dissatisfaction language, not just a negatively-phrased question) rather than
a technical safeguard; per the scoping conversation for this slice, a
distinguishable disable-reason or a staff notification are both explicitly
out of scope. This tradeoff is accepted, not solved, by this design.

## Architecture

Unchanged from the original design's shape — classification still runs
concurrently with RAG retrieval via `asyncio.gather`, and still layers into
the system prompt ahead of the RAG style block. The one new branch is the
`complaint` short-circuit in the webhook, inserted between that `gather` and
the existing `generate_reply` call:

```
app/webhook.py
   │  asyncio.gather(
   │      rag.retrieve(text, k),        # unchanged
   │      intent.classify(window),      # now returns one of six intents
   │  )
   ▼
   intent == COMPLAINT?
   │
   ├─ yes → admin_store.set_conversation_disabled(sender_id, True); no reply
   │
   └─ no  → app/llm.py generate_reply(turns, examples, intent)   # unchanged shape
              │  system prompt = SYSTEM_PROMPT
              │                + INTENT_ADDENDA[intent]   (if present)
              │                + style block               (if examples)
              ▼
            Anthropic reply generation (unchanged)
```

## Interfaces

### `app/intent.py`

`Intent` gains two members:

```python
class Intent(StrEnum):
    PRICE = "price"
    BOOKING = "booking"
    DESIGN = "design"
    AFTERCARE = "aftercare"
    COMPLAINT = "complaint"
    GENERAL = "general"
```

`CLASSIFY_SYSTEM_PROMPT` is rewritten to describe all six, with examples
drawn from real corpus phrasing where available:

```
- price: asking about cost, price, deposit, rate, or a discount/promo. E.g.
"Πόσο κοστίζει ένα μικρό τατουάζ;", "How much would this cost?", "τι τιμή
έχει;", "έχετε κάποια έκπτωση;"
- booking: asking about availability, wanting to schedule, reschedule, or
cancel an appointment, or asking about the booking process itself (payment
method, whether a deposit is required to reserve a slot, age/ID
requirements). E.g. "Έχετε ραντεβού αυτή την εβδομάδα;", "Can I book for next
Friday?", "θέλω να ακυρώσω το ραντεβού μου", "δέχεστε κάρτα;", "είμαι 17,
γίνεται;"
- design: describing or discussing a tattoo idea, placement, size, style, or
reference images. E.g. "Θέλω ένα μικρό τριαντάφυλλο στον καρπό", "I'm
thinking of a fine-line piece on my forearm"
- aftercare: asking about healing, aftercare instructions, or a possible
skin reaction after a tattoo. E.g. "Είναι φυσιολογικό να φαγουρίζει;", "How
long until it fully heals?", "μου κοκκίνισε γύρω γύρω, είναι εντάξει;"
- complaint: expressing dissatisfaction, a bad experience, or a problem with
a tattoo or the service received — not just a neutral question. E.g. "Δεν
είμαι καθόλου ευχαριστημένος με το αποτέλεσμα", "This isn't what I asked
for", "το τατουάζ μου χάλασε και δεν απαντάτε", "θέλω να κάνω παράπονο"
- general: anything else — greetings, thanks, small talk, studio-info
questions (hours, location, artist), or unclear messages.
```

The tool schema's `intent` enum is generated from `[i.value for i in Intent]`
already, so it picks up both new values with no code change beyond the enum
itself.

### `app/llm.py`

`INTENT_ADDENDA` gains one new entry and three revised ones; `COMPLAINT` is
intentionally absent (see Decisions):

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
    # generate_reply for a COMPLAINT-classified message (see app/webhook.py
    # below). If that short-circuit is ever bypassed, INTENT_ADDENDA.get()
    # falls back to no addendum — the same safe default as GENERAL — rather
    # than raising.
}
```

No other changes to `app/llm.py` — `generate_reply`'s signature, ordering
logic, and error handling are all unchanged from the original slice.

### `app/webhook.py`

One new branch inserted between the existing `gather` and the existing
`generate_reply` call:

```python
examples, msg_intent = await asyncio.gather(
    retrieve(text, settings.rag_top_k),
    classify(window),
)

if msg_intent is Intent.COMPLAINT:
    logger.info(
        "Auto-disabling conversation %s: classified as complaint", sender_id
    )
    await asyncio.to_thread(admin_store.set_conversation_disabled, sender_id, True)
    continue

reply = await generate_reply(window, examples, msg_intent)
```

Requires importing `Intent` alongside `classify` from `app.intent`. No other
change to the webhook: the inbound message is already persisted via `append`
earlier in the loop, before this branch, so a complaint is never lost even
though no reply is sent for it. The webhook's existing disabled-conversation
check (which runs before classification, at the top of the per-sender loop)
means a thread auto-disabled by this branch will short-circuit out of
processing entirely on the next inbound message, exactly as a manually
disabled conversation does today — no repeated classify/disable calls.

## Configuration

None. No new variables, no changes to `intent_model` or `intent_max_tokens` —
a six-value enum tool call is not meaningfully larger than a four-value one.

## Error Handling

| Failure | Behavior |
|---|---|
| Any `intent.classify` failure mode (API error, refusal, no tool_use block, unrecognized value) | Unchanged: returns `Intent.GENERAL`, same as before this slice |
| `complaint` classification is a false positive | **Accepted risk, not mitigated technically.** No reply is sent, and the conversation is disabled until a human notices via the dashboard. Mitigated only by classify-prompt example quality (Decisions). No push notification, no distinguishable disable-reason — both explicitly out of scope. |
| `admin_store.set_conversation_disabled` raises (e.g. DB unavailable) | Not caught specially by this slice — same unhandled-exception posture as the rest of the per-message loop body already has for `append`/`send_text` failures. A single message's processing may error, but the webhook still returns 200 to Meta since the exception is scoped inside the per-sender loop iteration, not the request handler. |
| Conversation already disabled when a second complaint-classified message arrives | No-op: `set_conversation_disabled(sender_id, True)` on an already-disabled conversation is idempotent. In practice this path is rarely reached — the earlier disabled-check already skips processing for a disabled conversation. |

## Testing

- `app/intent.py`: Greek + English fixtures for `aftercare` and `complaint`,
  drawn from real corpus phrasing where available. A fixture pair locking in
  the `price`/`booking` deposit split: "πόσο είναι η προκαταβολή;" (amount →
  `price`) vs. "χρειάζεται προκαταβολή για να κλείσω;" (process → `booking`).
  Existing fixtures and failure-mode tests (→ `Intent.GENERAL`) unchanged and
  still valid against the six-value enum.
- `app/llm.py`: addendum-presence assertions extended to `Intent.AFTERCARE`.
  New test asserting `generate_reply(turns, examples, Intent.COMPLAINT)`
  produces a system prompt **identical** to `Intent.GENERAL`/`None` — this is
  the defense-in-depth guarantee from Decisions, verified rather than just
  reasoned about. Existing addendum-ordering test unchanged.
- `app/webhook.py`: new test asserting that when `classify` returns
  `Intent.COMPLAINT`, `generate_reply` and `send_text` are **not** called and
  `admin_store.set_conversation_disabled` **is** called with
  `(sender_id, True)`. New test asserting the inbound message is still
  persisted via `append` on that path. Regression coverage confirming every
  other intent (including `AFTERCARE`) still flows through
  `generate_reply`/`send_text` unchanged. Existing concurrency test
  (`retrieve`/`classify` run via `asyncio.gather`) unchanged.
- No test makes a real Anthropic API call — same mocking pattern as the
  original slice and the rest of the suite.

## Container Changes

None. No new dependencies, no new files, no new bind mounts, no new config —
pure application code changes to three existing files.
