from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CANNED_REPLY = (
    "Γεια σου! Ελάβαμε το μήνυμά σου και θα σου απαντήσουμε σύντομα."
)


class Settings(BaseSettings):
    """Application configuration, validated once at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ig_verify_token: str
    ig_user_access_token: str
    ig_app_secret: str
    ig_account_id: str
    ig_api_version: str = "v26.0"
    port: int = 3000
    canned_reply: str = DEFAULT_CANNED_REPLY
    log_level: str = "INFO"

    # Comma-separated IG-scoped sender ids. While set, only these senders get
    # replies — everyone else's DMs are received and stored but ignored. This
    # is a temporary gate for testing before the assistant goes live for all
    # customers; leave empty to reply to everyone.
    allowed_sender_ids: str = ""

    @property
    def allowed_sender_id_set(self) -> frozenset[str]:
        return frozenset(
            sender_id.strip()
            for sender_id in self.allowed_sender_ids.split(",")
            if sender_id.strip()
        )

    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-5"
    llm_max_tokens: int = 2000
    llm_effort: str = "low"

    db_path: str = "./data/history.db"
    history_retention_days: int = 20
    history_window_messages: int = 20

    # Bearer token the admin API requires on every /admin/* request. Set by
    # the studio's website backend (Netlify function), not the browser.
    assistant_admin_key: str = ""

    # Style retrieval from past DM history (RAG). Embeddings go through
    # OpenRouter (openrouter.ai/api/v1/embeddings), not a direct Voyage AI
    # account. Leave OPENROUTER_API_KEY blank to keep the feature off — the
    # assistant behaves exactly as without it.
    openrouter_api_key: str = ""
    embedding_model: str = "voyageai/voyage-4"
    rag_top_k: int = 3
    rag_index_path: str = "./data/rag_index.json"

    # Intent classification (price/booking/design/aftercare/complaint/general),
    # run before every reply to steer its phrasing — or, for complaint, to
    # suppress it (see app/webhook.py). Deliberately a separate, cheaper/
    # faster model from ANTHROPIC_MODEL — this call happens on every message.
    intent_model: str = "claude-haiku-4-5-20251001"
    intent_max_tokens: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
