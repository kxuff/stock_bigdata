from typing import Any

from app.config import AgentSettings


class CrewAIUnavailableError(RuntimeError):
    """Raised when CrewAI is required but not installed."""


def build_deepseek_llm(
    settings: AgentSettings,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
) -> Any:
    try:
        from crewai import LLM
    except ModuleNotFoundError as exc:
        raise CrewAIUnavailableError("CrewAI is required to create a DeepSeek LLM") from exc

    resolved_model = f"deepseek/{settings.deepseek_model}" if model is None else model
    resolved_api_key = (
        api_key
        if api_key is not None
        else (
            settings.deepseek_api_key.get_secret_value()
            if settings.deepseek_api_key is not None
            else None
        )
    )
    resolved_base_url = settings.deepseek_base_url if base_url is None else base_url
    resolved_temperature = settings.agent_temperature if temperature is None else temperature
    resolved_timeout = settings.agent_timeout_seconds if timeout is None else timeout
    resolved_max_retries = settings.agent_max_retries if max_retries is None else max_retries

    try:
        return LLM(
            model=resolved_model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            temperature=resolved_temperature,
            timeout=resolved_timeout,
            max_retries=resolved_max_retries,
        )
    except TypeError:
        # Some CrewAI/LiteLLM versions expose the OpenAI-compatible endpoint as api_base.
        return LLM(
            model=resolved_model,
            api_key=resolved_api_key,
            api_base=resolved_base_url,
            temperature=resolved_temperature,
            timeout=resolved_timeout,
            max_retries=resolved_max_retries,
        )
