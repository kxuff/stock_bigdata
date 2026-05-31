import pytest
from pydantic import ValidationError

from app.config import AgentSettings, load_settings


def test_load_settings_uses_tool_result_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOOL_RESULT_PROVIDER", " BigData ")
    monkeypatch.delenv("ORCA_TOOL_RESULT_PROVIDER", raising=False)

    settings = load_settings(env_file=None)

    assert settings.tool_result_provider == "bigdata"


def test_load_settings_uses_orca_tool_result_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOOL_RESULT_PROVIDER", raising=False)
    monkeypatch.setenv("ORCA_TOOL_RESULT_PROVIDER", " bigdata ")

    settings = load_settings(env_file=None)

    assert settings.tool_result_provider == "bigdata"


def test_agent_settings_rejects_sample_tool_result_provider() -> None:
    with pytest.raises(ValidationError):
        AgentSettings(tool_result_provider="sample")


def test_agent_settings_rejects_unknown_tool_result_provider() -> None:
    with pytest.raises(ValidationError):
        AgentSettings(tool_result_provider="csv")


def test_load_settings_uses_generic_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", " openai ")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "generic-key")

    settings = load_settings(env_file=None)

    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_base_url == "https://example.test/v1"
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "generic-key"


def test_load_settings_uses_orca_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("ORCA_LLM_PROVIDER", "litellm")
    monkeypatch.setenv("ORCA_LLM_MODEL", "anthropic/claude-3-5-haiku")

    settings = load_settings(env_file=None)

    assert settings.llm_provider == "litellm"
    assert settings.llm_model == "anthropic/claude-3-5-haiku"


def test_agent_settings_rejects_unknown_llm_provider() -> None:
    with pytest.raises(ValidationError):
        AgentSettings(llm_provider="local")
