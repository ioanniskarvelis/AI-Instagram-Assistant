# Instagram Assistant — Style Retrieval from Past DMs (RAG)

**Date:** 2026-08-12
**Status:** Approved, ready for planning
**Builds on:** `docs/superpowers/specs/2026-08-11-conversation-history-design.md`

## Context

The v2 slice gave the assistant a hard-coded persona in `SYSTEM_PROMPT`, generic
enough to be safe but not distinctively "the studio's" voice. The studio has 3,341
past Instagram DM threads exported from Meta (`inbox/`, ~514MB, one folder per
thread), containing years of Christina's actual replies to customers.

This slice builds a retrieval corpus from that export and uses it to influence the
assistant's phrasing and tone at generation time — top-k similar past exchanges,
surfaced as a style reference, not as facts. It does not change what the assistant
is allowed to say; the existing hard rules (no prices, no bookings, no unverified
facts) stay exactly as strict as they are today. It changes how the assistant says
the things it's already allowed to say.

**Amendment to the v2 non-goal:** v2 explicitly deferred "vector retrieval of any
kind." This slice picks that up, scoped narrowly to style/tone influence rather
than answering questions from historical data as facts.

## Goals

1. Extract (question, reply) pairs from `inbox/`, reconstructing who-said-what per
   thread despite Meta's mojibake-encoded export text.
2. Give a human (the studio owner) a review checkpoint over the extracted corpus
   before anything is embedded or served.
3. At reply time, retrieve the top-k most similar past exchanges to the incoming
   DM and surface them to Claude as a style reference, influencing phrasing without
   ever overriding the existing hard content rules.
4. Never let stale historical prices, booking times, or other figures reach the
   model, even indirectly through a retrieved example.
5. Degrade to today's behavior (no retrieval) on any failure — missing index,
   embedding API down, empty corpus.

## Non-Goals

- Answering customer questions *from* historical data as facts (e.g. "what did we
  tell the last customer about our hours"). Retrieval only ever influences tone.
- Continuous/online corpus updates from the assistant's own future replies. The
  index is a static artifact, rebuilt manually when wanted.
- Image/attachment content from `inbox/*/photos/`. Text only.
- Any change to the existing pricing/booking/facts hard rules in `SYSTEM_PROMPT`.
- A vector database. Corpus size (low thousands of pairs at most, after filtering)
  fits a flat in-memory matrix with numpy cosine similarity.

## Decisions

### Pricing handling: scrub, don't drop

Exchanges that mention price, deposits, or booking times are **kept** in the
corpus — dropping them would remove some of the most common, most representative
customer interactions the assistant needs to sound natural for — but the studio's
reply text is scrubbed of currency amounts, numbers-adjacent-to-currency-words, and
booking-time phrasing, replaced with a neutral placeholder (`[price]`). This
preserves the phrasing and structure ("that depends on the design, send a photo
and we'll tell you...") without ever exposing an old, possibly wrong, number to the
model. The existing `SYSTEM_PROMPT` rule against quoting prices is unchanged and is
the actual enforcement; scrubbing is defense in depth against the model echoing a
retrieved figure verbatim.

Scrubbing is regex-based and imperfect by nature — it will miss some phrasing and
over-scrub others. That inexactness is why a human review step sits between
extraction and embedding.

### Human review checkpoint

Extraction writes candidate pairs to a git-ignored JSONL file. The studio owner
reviews and prunes it before the build step embeds anything. This is a manual,
one-time (or occasional) step outside the running application — there is no admin
UI for it.

### Embeddings: Voyage AI, not a local model

Query and corpus embeddings come from Voyage AI's hosted multilingual embedding
endpoint, called over `httpx` (no new SDK — same pattern as `app/instagram.py`).
This keeps the Docker image free of ML libraries/weights and fits the existing
architecture, which already makes a network call per message to Anthropic. The
cost of one more small network call is accepted in exchange for not bloating the
image or adding CPU-bound inference to request handling.

### Injection: labeled system-prompt block, not fake conversation turns

Retrieved examples are appended to the **system** prompt as a clearly labeled
"style reference" section, placed *after* the existing hard rules, never spliced
into the `messages` array as if they were prior turns of the current conversation.
Splicing them into `messages` risks the model treating another customer's design
idea or context as part of the current thread. A labeled, separate block keeps the
current conversation's `messages` array exactly what it is today — only this
customer's turns — while still giving Claude the stylistic signal.

### Index storage: read-only bind mount, not the named volume

`rag_index.json` is a small, static build artifact produced entirely outside the
container. It is bind-mounted read-only into the `api` service alongside the
existing named volume for `history.db`:

```yaml
volumes:
  - history:/srv/data
  - ./data/rag_index.json:/srv/data/rag_index.json:ro
```

Rebuilding the corpus means re-running the offline scripts and replacing that
file on the host — no image rebuild, no code change, no write access needed from
inside the container.

## Architecture

Three offline scripts (run on the host, using the existing `.venv`, not part of
the deployed image) and one new runtime module:

```
inbox/ (raw export, git-ignored, never touches the container)
   │  scripts/rag_extract.py
   ▼
data/rag_corpus_review.jsonl        (git-ignored)
   │  manual review by the studio owner
   ▼
data/rag_corpus_approved.jsonl      (git-ignored)
   │  scripts/rag_build_index.py  (calls Voyage AI)
   ▼
data/rag_index.json                 (git-ignored, bind-mounted read-only)
   │  app/rag.py  (loaded once at startup)
   ▼
app/webhook.py → app/llm.py         (top-k examples → system prompt block)
```

| Module | Responsibility | Depends on |
|---|---|---|
| `scripts/rag_extract.py` (new) | Walks `inbox/`, fixes mojibake, reconstructs turns, pairs customer→studio exchanges, filters/dedupes, scrubs prices, writes review JSONL. Run once per corpus refresh. | stdlib only |
| `scripts/rag_build_index.py` (new) | Reads the approved JSONL, embeds each customer question via Voyage AI, writes `rag_index.json`. Run once per corpus refresh. | `httpx` |
| `app/rag.py` (new) | Loads the index at startup; embeds an incoming message and returns its top-k similar (question, reply) pairs. The only runtime module that knows Voyage AI or the index format exist. | `config`, `httpx`, `numpy` |
| `app/llm.py` (modified) | `generate_reply` gains an `examples` parameter; builds the style-reference block when non-empty. | `rag` (only for the `Example` type) |
| `app/webhook.py` (modified) | Calls `rag.retrieve` before `generate_reply`, same never-block-the-reply posture as the rest of the flow. | `rag` |

## Data Model

### `rag_corpus_review.jsonl` / `rag_corpus_approved.jsonl`

One JSON object per line:

```json
{"thread_id": "1032708327539482", "customer": "Πόσο κοστίζει ένα μικρό τατουάζ;", "studio_reply_scrubbed": "Εξαρτάται από το σχέδιο και το μέγεθος — στείλε φωτο και σου λέμε [price] 🙏"}
```

`thread_id` is kept only to make review traceable back to the source folder if a
line looks wrong; it is dropped at the index-build step and never reaches the
running app.

### `rag_index.json`

```json
[
  {"question": "Πόσο κοστίζει ένα μικρό τατουάζ;",
   "reply": "Εξαρτάται από το σχέδιο και το μέγεθος — στείλε φωτο και σου λέμε [price] 🙏",
   "embedding": [0.0123, -0.0456, ...]}
]
```

Loaded once into a numpy matrix (`embedding` rows) plus a parallel list of
`(question, reply)` at process startup. No database, no on-disk index format
beyond this one JSON file.

## Interfaces

### `scripts/rag_extract.py`

Turn reconstruction, per thread:

1. Merge all `message_*.json` files in the thread (28 of the 3,341 threads span
   more than one), sort all messages by `timestamp_ms`.
2. Fix mojibake on `content` and participant `name`: `s.encode("latin1").decode("utf-8")`,
   falling back to the original string on `UnicodeError` (defensive; not expected
   to trigger given the export is consistently double-encoded).
3. Identify role by name match against the studio's participant name
   (`"2310tattoo studio by Christina"`, confirmed constant across all but ~50 of
   3,341 threads during exploration) — that participant's messages are `studio`,
   every other participant is `customer`.
4. Collapse consecutive same-role messages into one turn (newline-joined).
5. Emit a pair for every `customer` turn immediately followed by a `studio` turn.
   A thread with no customer reply (e.g. studio-only opening message) yields no
   pairs.
6. Drop a pair if either side is under ~3 characters after stripping emoji/
   punctuation, or if `studio_reply_scrubbed` is identical to one already emitted
   (dedupes the repeated opening greeting template).
7. Scrub the studio reply for price/currency/booking-time patterns (regex on `€`,
   "ευρώ"/"euro", digits adjacent to those, common Greek/English date-time
   phrasing) before dedup and before writing, replacing matches with `[price]`.
8. Append the surviving pair to `data/rag_corpus_review.jsonl`.

### `scripts/rag_build_index.py`

- Reads `data/rag_corpus_approved.jsonl` (the human-edited copy of the review
  file — the script does not read the review file directly, so an accidental
  re-run before review can't silently ship unreviewed data).
- Batches the `customer` field of every line through Voyage AI's embed endpoint.
- Writes `data/rag_index.json` in the shape above.

### `app/rag.py`

- `Example` — `question: str`, `reply: str` (dataclass, mirrors `history.Turn`'s
  role as a domain type, not a wire type)
- `async retrieve(text: str, k: int) -> list[Example]` — never raises; returns
  `[]` on any failure (index not loaded, Voyage API error, empty corpus). Embeds
  `text`, computes cosine similarity against the loaded matrix, returns the top
  `k` by score.
- Module-level load of `rag_index.json` at import time; a load failure logs a
  warning and leaves the module permanently in "no index" state rather than
  retrying per-request or crashing startup — this is reference/style data, not
  required for the app to serve replies correctly.

### `app/llm.py`

- `generate_reply(turns: list[Turn], examples: list[Example] | None = None) -> str | None`
- When `examples` is non-empty, the system block becomes `SYSTEM_PROMPT` +
  a rendered style-reference section (examples, each capped to a short length so
  `k=3` stays well within the existing 2000-token budget) + a one-line reminder
  that the rules above still apply. When `examples` is empty or `None`, the system
  block is exactly today's `SYSTEM_PROMPT` — no behavior change for a cold corpus
  or a retrieval failure.

## Data Flow

`POST /webhook`, per replyable message, extending the existing flow:

```
persist inbound turn → read window → retrieve style examples → generate reply → send → persist outbound turn
```

`rag.retrieve` is called with the incoming message text (not the full window) —
it's finding stylistically similar *questions*, not summarizing a conversation.
Its failure mode (`[]`) is silent and non-blocking, same posture as
`history.recent` returning `[]` today: the reply still goes out, just without the
style nudge.

## Configuration

Added to `app/config.py` / `.env`:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `VOYAGE_API_KEY` | yes, if RAG is used | — | Embedding API credential |
| `VOYAGE_MODEL` | no | `voyage-3.5` | Embedding model (multilingual, covers Greek + English) — confirm the current recommended multilingual model name against Voyage's docs during implementation; it is configuration, not a code change, if it differs |
| `RAG_TOP_K` | no | `3` | Examples retrieved per message |
| `RAG_INDEX_PATH` | no | `/srv/data/rag_index.json` | Where `app/rag.py` loads the built index from |

`VOYAGE_API_KEY` is not marked required at the `Settings` level the way
`ANTHROPIC_API_KEY` is — an empty key means `app/rag.py` never calls out and
`retrieve` always returns `[]`, so the assistant still runs (today's behavior)
without it configured.

## Error Handling

| Failure | Behavior |
|---|---|
| `rag_index.json` missing or unreadable at startup | Logged once; `retrieve` returns `[]` for the process lifetime |
| Voyage API error/timeout embedding a query | Logged; `retrieve` returns `[]` for that message only |
| Corpus empty after filtering/review | Same as missing index — `[]` |
| `VOYAGE_API_KEY` unset | `retrieve` short-circuits to `[]` without attempting a call |

No new failure path reaches the customer differently than today: a retrieval
failure is indistinguishable, from the webhook's point of view, from a cold
corpus. The existing `generate_reply` failure/fallback-to-`CANNED_REPLY` behavior
is unchanged.

## Testing

- `scripts/rag_extract.py`: mojibake fix round-trip, turn collapsing, pairing
  (including the "studio-only thread yields nothing" case and the multi-file
  thread merge/sort), dedup, and the price-scrub regex against representative
  Greek and English fixtures — this is the highest-value test surface since its
  output quality drives everything downstream and it only ever runs offline.
- `app/rag.py`: cosine similarity ranking and top-k ordering against a small
  in-memory fake index; `retrieve` returns `[]` on missing index, on a mocked
  Voyage failure (via `respx`, same pattern as the Graph API and Anthropic
  mocking already in the test suite), and on an unset API key.
- `app/llm.py`: system prompt contains the hard rules both with and without
  examples present, and — when present — the rules appear **after** the
  style-reference block, not before.
- No test depends on `inbox/` existing; it is git-ignored and absent in CI.

## Container Changes

- `compose.yaml` — add the read-only bind mount for `rag_index.json`
- `requirements.txt` — add `numpy`, version-pinned like every other entry
- `.env.example` — the four new variables above, `VOYAGE_API_KEY` blank
- `.gitignore` / `.dockerignore` — already updated to exclude `inbox/`; also
  exclude `data/rag_corpus_review.jsonl`, `data/rag_corpus_approved.jsonl`, and
  `data/rag_index.json` (all derived from customer data, none of it belongs in
  git)
