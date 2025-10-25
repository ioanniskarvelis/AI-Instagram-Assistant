# 210tattoo Assistant

![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000?logo=flask&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-queueing-dc382d?logo=redis&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)
![Google Calendar](https://img.shields.io/badge/Google%20Calendar-API-4285F4)
![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

Smart assistant that turns Instagram DMs into a streamlined booking flow for a tattoo studio. It classifies intent, retrieves similar past replies, checks real‑time availability, creates/reschedules/cancels Google Calendar events, and replies in Greek with the studio’s tone. Includes image‑assisted pricing via a vision prompt, Redis‑backed message queuing, and Pinecone‑powered retrieval for consistent, on‑brand responses.

## Demo
<details>
  <summary>Watch demo</summary>

  <a href="https://youtube.com/shorts/jegr-F_Fml0?si=cYtR2bduhi3mGFI8">Follow the link to watch the demo</a>
</details>

## Features
- Intent classification and response generation (OpenAI)
- Google Calendar availability checks and bookings
- Redis-backed message queue and slot holds
- Image analysis prompt workflow for pricing

## Frameworks & how they’re used
- Flask: HTTP server exposing `/`, `/health`, `/webhook`, `/privacy_policy`, `/terms_of_service` and wiring request handling to the assistant logic.
- Redis: transient storage for chat context (`chat:{user}`), message queue (`message_queue:{user}`), processing locks (`processing_lock:{user}`), mutes (`mute:{user}`), scheduling flags (`scheduled:{user}`) and temporary slot holds (`hold:YYYY-MM-DDTHH:MM`).
- OpenAI API: chat completions for replies, function/tool-calling for calendar actions, intent classification, and vision-assisted image analysis. Models are configurable via env.
- Pinecone: retrieval of similar past conversations and pricing examples using two indices (`tattoo-conversations`, `tattoo-pricing`) to keep responses consistent with the studio’s tone.
- Sentence-Transformers: generates embeddings (`paraphrase-multilingual-mpnet-base-v2`) for Pinecone semantic search.
- Google Calendar API: checks availability, creates/reschedules/cancels events, adds reminders, and handles timezone (Europe/Athens).
- Requests: calls Instagram Graph API to send DMs and downloads image attachments for analysis.
- python-dotenv: loads configuration from `.env` for local development.
- pytz: timezone-aware date math and formatting.
- Gunicorn: optional production WSGI server (see `requirements.txt`).

## Quickstart
1. Create and fill a `.env` from `.env.example`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run locally:
   ```bash
   python app.py
   ```

## Environment Variables
See `.env.example` for all variables. Important:
- `OPENAI_API_KEY`
- `IG_USER_ACCESS_TOKEN`
- `PINECONE_API_KEY`
- `REDIS_URL` (or host/port/user/pass)
- `GOOGLE_TOKEN_FILE` and `GOOGLE_CLIENT_SECRETS`

## Credentials
- Do not commit real tokens or OAuth credentials. Use `.gitignore` (already included).
- Generate `token.json` for Google Calendar locally and mount it when deploying.

## Disclaimer
This is a portfolio/demo application. Replace placeholder legal documents and review prompts before production use.
