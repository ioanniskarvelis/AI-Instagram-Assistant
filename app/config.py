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
    ig_api_version: str = "v22.0"
    port: int = 3000
    canned_reply: str = DEFAULT_CANNED_REPLY
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
