# Instagram Assistant v2 — Conversation History and LLM Replies

**Date:** 2026-08-11
**Status:** Approved, ready for planning
**Builds on:** `docs/superpowers/specs/2026-08-11-instagram-assistant-v1-design.md`

## Context

v1 delivered the plumbing slice: a containerized FastAPI service that receives
Instagram DM webhooks, verifies Meta's signature, filters echoes, and replies with
fixed text. It stores nothing.

This slice replaces the fixed reply with a Claude-generated one, and gives the
assistant enough memory to hold a coherent conversation across messages.

## Goals

1. Persist every turn of every conversation, surviving container restarts.
2. Generate replies with Claude, reading recent history so the assistant does not
   treat each DM as a cold start.
3. Delete stored messages automatically after 20 days.
4. Keep the storage layer behind an interface that the Telegram escalation slice
   can extend with a second table rather than a second store.

## Non-Goals

Deferred to later slices:

- Telegram Q&A escalation to the owner for pricing
- Google Calendar booking (booking state lives in Calendar, not here)
- Image-based tattoo quoting
- Message batching / grace window
- Any human-facing view of stored conversations
- Vector retrieval of any kind

## Decisions

### Storage: SQLite, not Postgres

The only thing that needs to persist is a 20-day rolling transcript for one studio,
at tens of messages a day, on a single always-on VPS.

Postgres was considered and rejected once two future slices stopped needing it:
booking state lives in Google Calendar and is reached through Calendar API calls,
and pricing is answered by a human over Telegram rather than by vector retrieval.
That left a single small relational table, which is squarely SQLite's job.

The tradeoff accepted: SQLite ties the app to one instance on one host. That is
consistent with the stated deployment target (a VPS or the studio's own machine
running Docker Compose). A move to a managed platform with ephemeral or unshared
local disk would require Postgres; the schema is small enough that this is a
migration, not a rewrite.

Redis was rejected because it is a poor host for the relational state the Telegram
slice will need, and running Redis *and* Postgres later would recreate the
multi-store sprawl this rebuild exists to remove.

### Retention: 20 days, time-based

Messages are deleted 20 days after they are written, regardless of conversation
activity. This covers a customer who enquires, thinks it over for a couple of
weeks, and comes back, without holding private DMs indefinitely.

Retention (what survives on disk) and the prompt window (what goes into the API
call) are independent settings and are configured separately.

### Model: Claude Opus 5

`claude-opus-5` via the `anthropic` SDK. The model ID is configuration, so moving
to Sonnet 5 or Haiku 4.5 for cost is an environment variable, not a code change.
Anthropic was chosen over OpenAI for instruction-following and persona adherence —
the assistant must reliably *not* invent prices.

### Amendment to the v1 dependency constraint

v1's spec froze runtime dependencies at exactly `fastapi`, `uvicorn`, `httpx`,
`pydantic-settings` and declared any addition out of scope. This slice adds
`anthropic` and nothing else. Storage uses the standard library's `sqlite3`.

## Architecture

Three new modules, one modified. `app/instagram.py` and `app/schemas.py` are
untouched.

| Module | Responsibility | Depends on |
|---|---|---|
| `app/db.py` (new) | Owns the SQLite connection and schema. Opens the file, sets pragmas, creates tables, runs the expiry sweep. The only module that knows SQLite exists. | `config` |
| `app/history.py` (new) | The conversation store as a domain interface. Speaks in turns, not rows. | `db` |
| `app/llm.py` (new) | The only module that knows the Anthropic API exists. | `config` |
| `app/webhook.py` (modified) | Orchestrates the five-step reply flow. | all of the above |

`webhook.py` contains no SQL and no Anthropic SDK calls. It reads as five named
steps. When the Telegram escalation slice arrives it becomes one more branch in
that orchestration, with a `pending_questions` table behind the same `db.py`.

### Async access to SQLite

The standard library's `sqlite3` is synchronous, and calling it directly from an
async handler blocks the event loop. The two store operations are wrapped in
`asyncio.to_thread` rather than adding an `aiosqlite` dependency. Writes are
single-row inserts and reads are one indexed query, so the thread-pool cost is
negligible.

**A connection is opened per operation, inside the worker thread**, rather than
sharing one process-wide connection. `asyncio.to_thread` runs each call on an
arbitrary thread-pool thread, and a `sqlite3.Connection` is bound to its creating
thread unless `check_same_thread=False` is passed — at which point correctness
depends on the underlying SQLite build's threading mode. Opening a connection to
an existing file is cheap enough at this volume that per-operation connections buy
that whole class of problem out of the design. `journal_mode=WAL` persists in the
database file and is set once at startup; `synchronous=NORMAL` is per-connection
and set on each open.

### Expiry enforcement

A `DELETE` sweep runs once at startup and then every six hours from a background
task. This is simpler and more predictable than pruning on every write, and the
sweep function takes an explicit `now` so it is testable without a time-mocking
library.

## Data Model

```sql
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    sender_id  TEXT    NOT NULL,
    role       TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    text       TEXT    NOT NULL,
    created_at INTEGER NOT NULL          -- Unix epoch seconds, UTC
);

CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages (sender_id, id);
CREATE INDEX IF NOT EXISTS idx_messages_expiry ON messages (created_at);
```

`role` uses `user` / `assistant` because those are exactly the values the Anthropic
messages array takes — the read path hands rows to the API with no translation
layer to keep in sync.

`created_at` is epoch seconds rather than an ISO string: no timezone ambiguity, and
the expiry sweep is an integer comparison. Readability in ad-hoc queries costs one
`datetime(created_at, 'unixepoch')`.

The two indexes serve the two access patterns: read the last N rows for one sender;
delete everything older than a cutoff across all senders.

Expected volume is roughly 2,000 rows at 100 messages/day over 20 days.

### Connection setup

WAL journal mode with `synchronous=NORMAL` — the standard pairing for a
single-writer local database. Durable across process crashes, and the expiry sweep
does not block a reply.

### File location and permissions

`DB_PATH` defaults to `./data/history.db` locally and `/srv/data/history.db` in the
container. The Dockerfile creates `/srv/data` owned by `appuser` (uid 1000) so the
named volume inherits that ownership on first mount.

### Privacy

Customer DM text is stored in plaintext on disk for 20 days. The sweep is the
deletion story. Encryption at rest for the VPS volume is a deployment decision,
noted here so it is made deliberately rather than by omission.

Logging is unchanged from v1: sender id and character count, never message content.

## Interfaces

### `app/db.py`

- `connect() -> sqlite3.Connection` — opens a connection to `DB_PATH` with
  `synchronous=NORMAL`; callers close it. Used as a context manager.
- `init_schema() -> None` — idempotent; sets `journal_mode=WAL` and creates the
  table and indexes. Called once at startup.
- `sweep_expired(now: int, retention_days: int) -> int` — returns rows deleted

### `app/history.py`

- `Turn` — `role: Literal["user", "assistant"]`, `text: str`
- `async append(sender_id: str, role: str, text: str) -> bool` — never raises;
  logs and returns `False` on failure
- `async recent(sender_id: str, limit: int) -> list[Turn]` — never raises; returns
  `[]` on failure. Returns turns oldest-first, with any **leading assistant turns
  dropped** so the list always begins on a `user` turn.

`Turn` is a domain type, not a wire type. `llm.py` maps each one to the API's
`{"role": ..., "content": ...}` shape; `history.py` has no knowledge of the
Anthropic message format beyond its choice of role names.

### `app/llm.py`

- `async generate_reply(turns: list[Turn]) -> str | None` — never raises; returns
  `None` on any failure

## Data Flow

`POST /webhook`, per replyable message:

```
persist inbound turn  →  read window  →  generate reply  →  send  →  persist outbound turn
```

The inbound turn is persisted *before* the window is read, so `recent()` returns a
window whose last element is the current message. That window is the entire input
to `generate_reply` — there is no separate "incoming message" parameter, and no
risk of the current turn appearing twice.

If `recent()` fails it returns `[]`, and the flow falls back to a single-turn window
containing just the incoming message. A storage failure costs conversational
memory, not the conversation.

### Two write-path rules

1. **Do not persist the echo.** Instagram delivers our outbound reply back to the
   webhook as `is_echo`. `WebhookPayload.replyable_messages()` already filters
   those, so the assistant turn is written exactly once — deliberately, by us.
2. **Persist the assistant turn only after a successful send.** If `send_text`
   returns `False` the customer never saw that message, and recording it would make
   the next prompt claim the studio said something it did not.

## The Claude Call

```python
response = await client.messages.create(
    model=settings.anthropic_model,          # "claude-opus-5"
    max_tokens=settings.llm_max_tokens,      # 2000
    system=[{"type": "text", "text": SYSTEM_PROMPT,
             "cache_control": {"type": "ephemeral"}}],
    output_config={"effort": settings.llm_effort},   # "low"
    messages=[{"role": t.role, "content": t.text} for t in turns],
)
```

**Adaptive thinking stays on at `effort: "low"`.** Thinking is on by default on
Opus 5 and counts against `max_tokens`, which is why the budget is 2000 rather than
500 for a two-sentence reply. Explicitly disabling thinking is the wrong lever: on
Opus 5 it can emit a tool call as plain text or leak `<thinking>` tags into the
visible response, and the documented fix is to leave thinking on and lower effort.
`low` also suits the task — a short conversational DM is not intelligence-sensitive.

**Prompt caching** is marked on the system prompt. It pays off during a live
back-and-forth and will usually miss on the first message of a conversation, since
the default TTL is five minutes and studio DMs arrive in bursts hours apart.
Scheduled pre-warming is explicitly not adopted: a cache write every five minutes
forever is not worth the latency saved on a handful of messages a day. Note that
Opus 5's minimum cacheable prefix is 512 tokens — a shorter system prompt silently
does not cache, reporting `cache_creation_input_tokens: 0` with no error.

### System prompt

The highest-risk component in this slice. Storage and API plumbing are mechanical;
an assistant that invents a price for a full sleeve is a real problem for the
studio. The prompt must state what the assistant does **not** do:

- Never quote, estimate, or range a price. Pricing belongs to the Telegram
  escalation slice; until then, say the artist will follow up.
- Never confirm, offer, or imply an appointment slot, and never claim availability.
  Booking belongs to the Calendar slice.
- Never invent studio facts (hours, artists, styles) not given in the prompt.

It must also reply in the customer's language, defaulting to Greek, and keep replies
to a couple of sentences.

Instagram's Graph API rejects overly long DM text. The exact character cap must be
confirmed against Meta's current documentation during implementation and enforced
before sending, rather than trusting the model to stay under it.

### Failure modes of `generate_reply`

Returns `None`, never raises, matching `instagram.send_text`:

- Transport or API errors: `APIConnectionError`, `RateLimitError`, `APIStatusError`
- `stop_reason == "refusal"` — Opus 5's classifiers can decline, arriving as a
  normal `200` with empty or partial content. **Checked before reading
  `response.content`**, which would otherwise raise on an empty list.
- `stop_reason == "max_tokens"` — a reply truncated mid-sentence is worse for a
  customer than a generic acknowledgement. Logged so the ceiling can be raised.

On `None`, the flow sends `CANNED_REPLY`. v1's fixed reply stops being placeholder
code and becomes the degradation path: the customer gets an acknowledgement instead
of silence, and Meta still gets its `200`.

## Configuration

Added to the v1 environment variables:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | yes | — | Claude API credential |
| `ANTHROPIC_MODEL` | no | `claude-opus-5` | Reply model |
| `LLM_MAX_TOKENS` | no | `2000` | Output ceiling (thinking + text) |
| `LLM_EFFORT` | no | `low` | Effort level |
| `DB_PATH` | no | `./data/history.db` | SQLite file location |
| `HISTORY_RETENTION_DAYS` | no | `20` | Days before a message is deleted |
| `HISTORY_WINDOW_MESSAGES` | no | `20` | Turns sent to the model |

`CANNED_REPLY` is unchanged and retains its v1 default.

## Error Handling

`POST /webhook` returns `200` to Meta unless the request itself was bad, so Meta
never retry-storms a request that will fail identically.

| Failure | Behavior | To Meta |
|---|---|---|
| Invalid / missing signature | Rejected before parsing | `403` |
| Malformed payload | Rejected | `400` |
| `history.append` fails | Logged, non-fatal; reply still sent | `200` |
| `history.recent` fails | Logged; falls back to single-turn window | `200` |
| `generate_reply` returns `None` | `CANNED_REPLY` sent instead | `200` |
| `send_text` returns `False` | Logged; no assistant turn persisted | `200` |
| Expiry sweep fails | Logged; retried on the next six-hour tick | n/a |

**Fail loudly at startup, degrade quietly at runtime.** If the database cannot be
opened at boot — volume not mounted, directory not writable — the container crashes
so the misconfiguration is caught at deploy time. Once serving, a runtime storage
error is treated as transient and the bot keeps answering with a degraded window.

## Testing

No new test dependencies. `respx` intercepts `httpx`, and the Anthropic SDK is built
on `httpx`, so calls to `api.anthropic.com` mock the same way the Graph API calls
already do.

Two design-for-test choices replace new tooling:

- `sweep_expired` takes an explicit `now`, so testing 20-day expiry is arithmetic
  rather than time mocking.
- `conftest` points `DB_PATH` at `tmp_path`, giving each test a real SQLite file it
  owns. Tests exercise actual SQLite, not a mock of it.

Coverage per module:

- **`db`** — schema creates idempotently; WAL enabled; sweep deletes past the cutoff
  and leaves the rest.
- **`history`** — append/read round-trip; window caps at N; leading assistant turns
  dropped; unknown sender returns `[]`; senders isolated from each other.
- **`llm`** — request carries model, system prompt, and full window; returns text on
  success; returns `None` on API error, transport error, `refusal`, and `max_tokens`.
- **`webhook`** — a DM stores two turns and sends the generated reply; an LLM failure
  sends `CANNED_REPLY` and returns `200`; a send failure stores no assistant turn; a
  second DM from the same sender includes the first exchange in the prompt; an echo
  stores nothing and sends nothing.

The echo test is the regression guard for the reply-loop bug. The second-DM test is
the one that proves the feature works; the rest is plumbing.

## Container Changes

- `Dockerfile` — create `/srv/data` owned by `appuser` before `USER appuser`
- `compose.yaml` — named volume mounted at `/srv/data` on the `api` service
- `.env.example` — the seven new variables above, with `ANTHROPIC_API_KEY` blank
- `requirements.txt` — add `anthropic`, version-pinned like every other entry

The image must stay under the 300MB limit from v1 (currently 53.2MB).
