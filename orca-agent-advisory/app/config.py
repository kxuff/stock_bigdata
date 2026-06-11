import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_provider: str = "litellm"
    llm_model: str = "openai/oc/deepseek-v4-flash-free"
    llm_base_url: str = "http://localhost:20128/v1"
    llm_api_key: SecretStr | None = None
    agent_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    agent_max_retries: int = Field(default=3, ge=0)
    agent_timeout_seconds: int = Field(default=180, ge=1)
    advisory_output_dir: Path = Path("outputs/advisory_decisions")
    decision_job_database_url: str | None = None
    decision_job_table: str = "orca_decision_jobs"
    agent_route_audit_database_url: str | None = None
    agent_route_audit_table: str = "orca_agent_route_audits"
    redis_url: str | None = None
    decision_job_queue: str = "orca-decision-jobs"
    crewai_verbose: bool = True
    crewai_tracing: bool = True
    crewai_share_crew: bool = True
    tool_result_provider: str = "bigdata"
    kafka_bootstrap_servers: str | None = None
    kafka_allowed_topics: list[str] = Field(default_factory=list)
    kafka_consumer_group: str | None = None
    kafka_inspection_timeout_seconds: float = Field(default=5.0, gt=0)
    kafka_sample_enabled: bool = True
    kafka_sample_max_bytes: int = Field(default=512, ge=0, le=4096)

    @field_validator("llm_provider")
    @classmethod
    def normalize_llm_provider(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned != "litellm":
            raise ValueError("llm_provider must be: litellm")
        return cleaned

    @field_validator("llm_model", "llm_base_url")
    @classmethod
    def normalize_llm_string(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value cannot be blank")
        return cleaned

    @field_validator("advisory_output_dir")
    @classmethod
    def normalize_output_dir(cls, value: Path) -> Path:
        return Path(value)

    @field_validator("tool_result_provider")
    @classmethod
    def normalize_tool_result_provider(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned != "bigdata":
            raise ValueError("tool_result_provider must be: bigdata")
        return cleaned


def load_settings() -> AgentSettings:
    return AgentSettings(
        llm_base_url=_read_env("LLM_BASE_URL") or AgentSettings.model_fields["llm_base_url"].default,
        llm_api_key=_read_secret_env("NINEROUTER_KEY"),
        agent_temperature=_read_float_env("AGENT_TEMPERATURE") or AgentSettings.model_fields["agent_temperature"].default,
        agent_max_retries=_read_int_env("AGENT_MAX_RETRIES") or AgentSettings.model_fields["agent_max_retries"].default,
        agent_timeout_seconds=_read_int_env("AGENT_TIMEOUT_SECONDS") or AgentSettings.model_fields["agent_timeout_seconds"].default,
        decision_job_database_url=_read_env("ORCA_DECISION_JOB_DATABASE_URL"),
        decision_job_queue=_read_env("ORCA_DECISION_JOB_QUEUE") or AgentSettings.model_fields["decision_job_queue"].default,
        agent_route_audit_database_url=_read_env("ORCA_AGENT_ROUTE_AUDIT_DATABASE_URL"),
        redis_url=_read_env("ORCA_REDIS_URL"),
        crewai_verbose=_read_bool_env("CREWAI_VERBOSE", default=AgentSettings.model_fields["crewai_verbose"].default),
        crewai_tracing=_read_bool_env("CREWAI_TRACING_ENABLED", default=AgentSettings.model_fields["crewai_tracing"].default),
        crewai_share_crew=_read_bool_env("CREWAI_SHARE_CREW", default=AgentSettings.model_fields["crewai_share_crew"].default),
    )


def _read_env(key: str) -> str | None:
    return os.getenv(key)


def _read_secret_env(key: str) -> SecretStr | None:
    value = _read_env(key)
    return SecretStr(value) if value is not None else None


def _read_bool_env(key: str, *, default: bool) -> bool:
    value = _read_env(key)
    if value is None:
        return default
    cleaned = value.strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean value")


def _read_float_env(key: str) -> float | None:
    value = _read_env(key)
    if value is None:
        return None
    return float(value)


def _read_int_env(key: str) -> int | None:
    value = _read_env(key)
    if value is None:
        return None
    return int(value)
