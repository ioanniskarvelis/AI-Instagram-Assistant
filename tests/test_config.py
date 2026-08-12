import pytest
from pydantic import ValidationError


def test_settings_read_from_environment():
    from app.config import get_settings

    settings = get_settings()
    assert settings.ig_verify_token == "test-verify-token"
    assert settings.ig_app_secret == "test-app-secret"


def test_settings_apply_documented_defaults():
    from app.config import get_settings

    settings = get_settings()
    assert settings.ig_api_version == "v26.0"
    assert settings.port == 3000
    assert settings.log_level == "INFO"


def test_missing_required_setting_raises(monkeypatch):
    from app.config import Settings, get_settings

    monkeypatch.delenv("IG_APP_SECRET", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_new_settings_apply_documented_defaults():
    from app.config import get_settings

    settings = get_settings()
    assert settings.anthropic_model == "claude-sonnet-5"
    assert settings.llm_max_tokens == 2000
    assert settings.llm_effort == "low"
    assert settings.history_retention_days == 20
    assert settings.history_window_messages == 20


def test_missing_anthropic_key_raises(monkeypatch):
    from app.config import Settings, get_settings

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_rag_settings_apply_documented_defaults():
    from app.config import get_settings

    settings = get_settings()
    assert settings.openrouter_api_key == ""
    assert settings.embedding_model == "voyageai/voyage-4"
    assert settings.rag_top_k == 3
    assert settings.rag_index_path.endswith("rag_index.json")
