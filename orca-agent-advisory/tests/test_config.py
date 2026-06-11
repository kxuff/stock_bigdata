import pytest
from pydantic import ValidationError

from app.config import AgentSettings, load_settings


def test_load_settings_ignores_tool_result_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOOL_RESULT_PROVIDER", " BigData ")
    monkeypatch.delenv("ORCA_TOOL_RESULT_PROVIDER", raising=False)

    settings = load_settings()

    assert settings.tool_result_provider == "bigdata"


def test_load_settings_ignores_orca_tool_result_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOOL_RESULT_PROVIDER", raising=False)
    monkeypatch.setenv("ORCA_TOOL_RESULT_PROVIDER", " csv ")

    settings = load_settings()

    assert settings.tool_result_provider == "bigdata"


def test_agent_settings_rejects_sample_tool_result_provider() -> None:
    with pytest.raises(ValidationError):
        AgentSettings(tool_result_provider="sample")


def test_agent_settings_rejects_unknown_tool_result_provider() -> None:
    with pytest.raises(ValidationError):
        AgentSettings(tool_result_provider="csv")


def test_load_settings_uses_only_ninerouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NINEROUTER_KEY", "ninerouter-key")
    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("ORCA_LLM_API_KEY", "orca-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    settings = load_settings()

    assert settings.llm_provider == "litellm"
    assert settings.llm_model == "openai/oc/deepseek-v4-flash-free"
    assert settings.llm_base_url == "http://localhost:20128/v1"
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "ninerouter-key"


def test_load_settings_uses_docker_llm_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.delenv("NINEROUTER_KEY", raising=False)

    settings = load_settings()

    assert settings.llm_provider == "litellm"
    assert settings.llm_model == "openai/oc/deepseek-v4-flash-free"
    assert settings.llm_base_url == "https://example.test/v1"
    assert settings.llm_api_key is None


def test_load_settings_uses_docker_job_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCA_REDIS_URL", "redis://redis:6379/0")
    monkeypatch.setenv("ORCA_DECISION_JOB_DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/stock_db")
    monkeypatch.setenv("ORCA_DECISION_JOB_QUEUE", "custom-decision-jobs")

    settings = load_settings()

    assert settings.redis_url == "redis://redis:6379/0"
    assert settings.decision_job_database_url == "postgresql://postgres:postgres@postgres:5432/stock_db"
    assert settings.decision_job_queue == "custom-decision-jobs"


def test_load_settings_uses_docker_runtime_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_TEMPERATURE", "0.4")
    monkeypatch.setenv("AGENT_MAX_RETRIES", "5")
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "240")
    monkeypatch.setenv("CREWAI_VERBOSE", "false")
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    monkeypatch.setenv("CREWAI_SHARE_CREW", "false")

    settings = load_settings()

    assert settings.agent_temperature == 0.4
    assert settings.agent_max_retries == 5
    assert settings.agent_timeout_seconds == 240
    assert settings.crewai_verbose is False
    assert settings.crewai_tracing is False
    assert settings.crewai_share_crew is False


def test_load_settings_ignores_old_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ORCA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("NINEROUTER_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    settings = load_settings()

    assert settings.llm_api_key is None


def test_config_flags_default_true() -> None:
    settings = AgentSettings()

    assert settings.crewai_verbose is True
    assert settings.crewai_tracing is True
    assert settings.crewai_share_crew is True
    assert settings.kafka_sample_enabled is True


def test_agent_settings_rejects_unknown_llm_provider() -> None:
    with pytest.raises(ValidationError):
        AgentSettings(llm_provider="local")
