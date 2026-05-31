import os
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    llm_provider: str = "deepseek"
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    agent_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    agent_max_retries: int = Field(default=3, ge=0)
    agent_timeout_seconds: int = Field(default=180, ge=1)
    advisory_use_crewai_manager: bool = True
    advisory_enable_critic_stage: bool = False
    advisory_use_llm_critic_stage: bool = False
    advisory_llm_critic_timeout_seconds: int = Field(default=60, ge=1)
    advisory_output_dir: Path = Path("outputs/advisory_decisions")
    decision_job_database_url: str | None = None
    decision_job_table: str = "orca_decision_jobs"
    redis_url: str | None = None
    decision_job_queue: str = "orca-decision-jobs"
    crewai_verbose: bool = False
    crewai_tracing: bool = True
    crewai_share_crew: bool = False
    tool_result_provider: str = "bigdata"

    @field_validator("deepseek_base_url", "deepseek_model")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value cannot be blank")
        return cleaned

    @field_validator("llm_provider")
    @classmethod
    def normalize_llm_provider(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in {"deepseek", "openai", "openai_compatible", "litellm"}:
            raise ValueError("llm_provider must be one of: deepseek, openai, openai_compatible, litellm")
        return cleaned

    @field_validator("llm_model", "llm_base_url")
    @classmethod
    def normalize_optional_llm_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def apply_deepseek_llm_defaults(self) -> "AgentSettings":
        if self.llm_provider == "deepseek":
            if self.llm_model is None:
                self.llm_model = self.deepseek_model
            if self.llm_base_url is None:
                self.llm_base_url = self.deepseek_base_url
            if self.llm_api_key is None:
                self.llm_api_key = self.deepseek_api_key
        return self

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


def load_settings(env_file: str | Path | None = ".env") -> AgentSettings:
    values = _load_env_file(env_file) if env_file else {}

    return AgentSettings(
        deepseek_api_key=_read_env("DEEPSEEK_API_KEY", values),
        deepseek_base_url=_read_env("DEEPSEEK_BASE_URL", values)
        or AgentSettings.model_fields["deepseek_base_url"].default,
        deepseek_model=_read_env("DEEPSEEK_MODEL", values)
        or AgentSettings.model_fields["deepseek_model"].default,
        llm_provider=(
            _read_env("LLM_PROVIDER", values)
            or _read_env("ORCA_LLM_PROVIDER", values)
            or AgentSettings.model_fields["llm_provider"].default
        ),
        llm_model=_read_env("LLM_MODEL", values) or _read_env("ORCA_LLM_MODEL", values),
        llm_base_url=_read_env("LLM_BASE_URL", values) or _read_env("ORCA_LLM_BASE_URL", values),
        llm_api_key=_read_env("LLM_API_KEY", values) or _read_env("ORCA_LLM_API_KEY", values),
        agent_temperature=float(
            _read_env("AGENT_TEMPERATURE", values)
            or AgentSettings.model_fields["agent_temperature"].default
        ),
        agent_max_retries=int(
            _read_env("AGENT_MAX_RETRIES", values)
            or AgentSettings.model_fields["agent_max_retries"].default
        ),
        agent_timeout_seconds=int(
            _read_env("AGENT_TIMEOUT_SECONDS", values)
            or AgentSettings.model_fields["agent_timeout_seconds"].default
        ),
        advisory_use_crewai_manager=_read_bool_env(
            "ADVISORY_USE_CREWAI_MANAGER",
            values,
            default=AgentSettings.model_fields["advisory_use_crewai_manager"].default,
        ),
        advisory_enable_critic_stage=_read_bool_env(
            "ADVISORY_ENABLE_CRITIC_STAGE",
            values,
            default=AgentSettings.model_fields["advisory_enable_critic_stage"].default,
        ),
        advisory_use_llm_critic_stage=_read_bool_env(
            "ADVISORY_USE_LLM_CRITIC_STAGE",
            values,
            default=AgentSettings.model_fields["advisory_use_llm_critic_stage"].default,
        ),
        advisory_llm_critic_timeout_seconds=int(
            _read_env("ADVISORY_LLM_CRITIC_TIMEOUT_SECONDS", values)
            or AgentSettings.model_fields["advisory_llm_critic_timeout_seconds"].default
        ),
        advisory_output_dir=Path(
            _read_env("ADVISORY_OUTPUT_DIR", values)
            or AgentSettings.model_fields["advisory_output_dir"].default
        ),
        decision_job_database_url=_read_env("DECISION_JOB_DATABASE_URL", values)
        or _read_env("ORCA_DECISION_JOB_DATABASE_URL", values),
        decision_job_table=(
            _read_env("DECISION_JOB_TABLE", values)
            or _read_env("ORCA_DECISION_JOB_TABLE", values)
            or AgentSettings.model_fields["decision_job_table"].default
        ),
        redis_url=_read_env("REDIS_URL", values) or _read_env("ORCA_REDIS_URL", values),
        decision_job_queue=(
            _read_env("DECISION_JOB_QUEUE", values)
            or _read_env("ORCA_DECISION_JOB_QUEUE", values)
            or AgentSettings.model_fields["decision_job_queue"].default
        ),
        crewai_verbose=_read_bool_env(
            "CREWAI_VERBOSE",
            values,
            default=AgentSettings.model_fields["crewai_verbose"].default,
        ),
        crewai_tracing=_read_bool_env(
            "CREWAI_TRACING_ENABLED",
            values,
            default=AgentSettings.model_fields["crewai_tracing"].default,
        ),
        crewai_share_crew=_read_bool_env(
            "CREWAI_SHARE_CREW",
            values,
            default=AgentSettings.model_fields["crewai_share_crew"].default,
        ),
        tool_result_provider=(
            _read_env("TOOL_RESULT_PROVIDER", values)
            or _read_env("ORCA_TOOL_RESULT_PROVIDER", values)
            or AgentSettings.model_fields["tool_result_provider"].default
        ),
    )


def _read_env(key: str, env_file_values: Mapping[str, str]) -> str | None:
    value = os.getenv(key)
    if value is not None:
        return value
    return env_file_values.get(key)


def _read_bool_env(key: str, env_file_values: Mapping[str, str], *, default: bool) -> bool:
    value = _read_env(key, env_file_values)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_env_file(env_file: str | Path | None) -> dict[str, str]:
    if env_file is None:
        return {}

    path = Path(env_file)
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value

    return values
