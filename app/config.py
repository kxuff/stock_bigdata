import os
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    agent_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    agent_max_retries: int = Field(default=3, ge=0)
    agent_timeout_seconds: int = Field(default=180, ge=1)

    @field_validator("deepseek_base_url", "deepseek_model")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value cannot be blank")
        return cleaned


def load_settings(env_file: str | Path | None = ".env") -> AgentSettings:
    values = _load_env_file(env_file) if env_file else {}

    return AgentSettings(
        deepseek_api_key=_read_env("DEEPSEEK_API_KEY", values),
        deepseek_base_url=_read_env("DEEPSEEK_BASE_URL", values)
        or AgentSettings.model_fields["deepseek_base_url"].default,
        deepseek_model=_read_env("DEEPSEEK_MODEL", values)
        or AgentSettings.model_fields["deepseek_model"].default,
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
    )


def _read_env(key: str, env_file_values: Mapping[str, str]) -> str | None:
    value = os.getenv(key)
    if value is not None:
        return value
    return env_file_values.get(key)


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
