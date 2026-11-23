"""Tests for centralized environment-backed settings."""

from video_edit_agent.config import DEFAULT_OLLAMA_MODEL, SETTING_DEFAULTS, get_setting


def test_ollama_model_defaults_to_documented_small_model(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert get_setting("OLLAMA_MODEL") == "qwen2.5vl:3b"
    assert get_setting("OLLAMA_MODEL") == DEFAULT_OLLAMA_MODEL


def test_environment_value_overrides_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "custom:tag")
    assert get_setting("OLLAMA_MODEL") == "custom:tag"


def test_environment_settings_and_defaults_are_catalogued():
    assert SETTING_DEFAULTS["OLLAMA_MODEL"] == "qwen2.5vl:3b"
    assert SETTING_DEFAULTS["OLLAMA_BASE_URL"] == "http://127.0.0.1:11434"
    assert SETTING_DEFAULTS["OLLAMA_KEEP_ALIVE"] == "5m"
