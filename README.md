# Tattoo Studio Instagram Assistant

Replies to Instagram DMs for the studio using Claude, remembering the last
20 messages of each conversation, and requesting tattoo price quotes from the
artists over Telegram. Calendar booking arrives in a later slice.

## Running locally

```bash
cp .env.example .env   # then fill in IG_USER_ACCESS_TOKEN, IG_APP_SECRET, and CLOUDFLARE_TUNNEL_TOKEN
docker compose --profile dev up --build
```

`CLOUDFLARE_TUNNEL_TOKEN` comes from a named Cloudflare Tunnel, set up once:

1. In the [Cloudflare Zero Trust dashboard](https://one.dash.cloudflare.com/),
   go to **Networks > Tunnels > Create a tunnel**, choose **Docker**, and copy
   the token from the generated run command (the `--token ...` value) into
   `CLOUDFLARE_TUNNEL_TOKEN`.
2. On the tunnel's **Public Hostname** tab, add a hostname (e.g.
   `webhook.yourdomain.com`) pointing to service `HTTP` → `api:3000`. This
   requires a domain on your Cloudflare account.

That hostname is permanent — it survives container restarts, so the Meta app
dashboard only needs to be configured once:

- Callback URL: `https://<your hostname>/webhook`
- Verify token: the value of `IG_VERIFY_TOKEN` in your `.env`

Subscribe to the `messages` field, then DM the studio account from a different
Instagram account. You should receive `CANNED_REPLY` back.

## Running the tests

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest -v
```

## Configuration

See `.env.example` for every supported variable and its default.

## Conversation storage

Conversations are stored in SQLite, in a Docker named volume mounted at
`/srv/data`. Messages are deleted automatically 20 days after they are written;
a sweep runs at startup and every six hours thereafter.

Two independent settings control this:

- `HISTORY_RETENTION_DAYS` — how long a message survives on disk
- `HISTORY_WINDOW_MESSAGES` — how many recent turns are sent to the model

Customer message text is stored in plaintext for the retention period. If that
matters for your deployment, encrypt the volume at rest.

To count or clear stored conversations:

```bash
docker compose exec api python -c "import sqlite3; print(sqlite3.connect('/srv/data/history.db').execute('SELECT COUNT(*) FROM messages').fetchone()[0])"
docker compose down -v   # removes the volume and all stored conversations
```

## Style retrieval from past DMs (RAG)

The assistant can draw on the studio's own past Instagram DM replies (a
Meta data export placed at `inbox/` in the project root, never committed) to
match its phrasing, without ever exposing an old price or booking time. This
is optional — leave `OPENROUTER_API_KEY` unset and the assistant behaves
exactly as it does without it. Embeddings go through
[OpenRouter](https://openrouter.ai/) (`voyageai/voyage-4`), not a direct
Voyage AI account.

Before your first `docker compose up`, create a placeholder index file so
Docker never turns the bind-mount path into a directory (see below):

```powershell
'[]' | Set-Content -Encoding utf8 data\rag_index.json
```

This must be a real *file*, not a directory. If `data/rag_index.json`
already exists as a directory (from an earlier `docker compose up` before
this file existed), delete it first — `Remove-Item -Recurse -Force
data\rag_index.json` — then create the placeholder above.  `app/rag.py`'s
`_load_index` already handles an empty/`[]` corpus gracefully, degrading to
no style examples.

Both scripts below use paths relative to the current working directory
(`inbox/`, `data/...`), so run them from the repository root.

To build or refresh the corpus:

```powershell
# 1. Extract candidate (question, reply) pairs from inbox/
.venv\Scripts\python -m scripts.rag_extract
# writes data/rag_corpus_review.jsonl

# 2. Review it. Copy your edited version to data/rag_corpus_approved.jsonl —
#    only what you approve here ever reaches the model.
Copy-Item data\rag_corpus_review.jsonl data\rag_corpus_approved.jsonl
# ... edit data/rag_corpus_approved.jsonl by hand ...

# 3. Embed the approved corpus via OpenRouter and write the runtime index
$env:OPENROUTER_API_KEY="..."; .venv\Scripts\python -m scripts.rag_build_index
# writes data/rag_index.json
```

`compose.yaml` bind-mounts `data/rag_index.json` read-only into the `api`
container. Refreshing the corpus means re-running steps 1–3 and restarting
the container (`docker compose restart api`) — no image rebuild needed.

Two related facts worth knowing about that bind mount: if
`data/rag_index.json` doesn't exist on the host at all when Docker starts
the container, Docker silently creates it as an empty *directory* rather
than failing — which then breaks `scripts/rag_build_index.py`'s attempt to
write the index file (`IsADirectoryError`). That's why the placeholder step
above matters. Separately, if the mounted file legitimately contains no
examples yet (e.g. the placeholder `[]`), the assistant degrades safely and
simply runs without style examples until you build a real index.

## Tattoo quoting via Telegram

Once the model has, for each piece a customer describes, the style, a
reference image or description, and size/placement, it calls a
`request_quote` tool. That sends the gathered details — plus every
reference-image URL the sender has sent so far — to the artists' Telegram
chat as one message. This is optional: leave `TELEGRAM_BOT_TOKEN` or
`TELEGRAM_CHAT_ID` unset and `request_quote` is never offered to the model,
same posture as `OPENROUTER_API_KEY` for RAG.

An artist replies with the price by using Telegram's native **Reply**
feature directly on that message — not just posting a new message in the
chat. `app/telegram_webhook.py` correlates the reply back to the right
Instagram sender via the message it replied to, phrases the price for the
customer in the assistant's usual tone, and sends it as a normal Instagram
DM.

Setup, once you have a bot (via [@BotFather](https://t.me/BotFather)) and
have added it to the artists' group:

1. Get the chat id: post any message in the group, then hit
   `https://api.telegram.org/bot<token>/getUpdates` and read `chat.id` off
   the response (it's negative for a group).
2. Fill in `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, and a random
   `TELEGRAM_WEBHOOK_SECRET` of your choosing in `.env`.
3. Point Telegram at the same tunnel hostname the Instagram webhook uses:
   ```bash
   curl "https://api.telegram.org/bot<token>/setWebhook?url=https://<your tunnel hostname>/telegram-webhook&secret_token=<TELEGRAM_WEBHOOK_SECRET>"
   ```

Reference images are passed to Telegram as Instagram's own CDN URLs rather
than downloaded and re-uploaded — Telegram fetches the URL server-side.
Those URLs are public and time-limited, so they're only ever forwarded live,
never fetched or stored by this app.
