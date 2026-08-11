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


@lru_cache
def get_settings() -> Settings:
    return Settings()
