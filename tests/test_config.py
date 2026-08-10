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
    assert settings.ig_api_version == "v22.0"
    assert settings.port == 3000
    assert settings.log_level == "INFO"


def test_missing_required_setting_raises(monkeypatch):
    from app.config import Settings, get_settings

    monkeypatch.delenv("IG_APP_SECRET", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
