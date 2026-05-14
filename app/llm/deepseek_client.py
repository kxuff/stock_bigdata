from typing import Any

from app.config import AgentSettings


class CrewAIUnavailableError(RuntimeError):
    """Raised when CrewAI is required but not installed."""


def build_deepseek_llm(settings: AgentSettings, **overrides: Any) -> Any:
    try:
        from crewai import LLM
    except ModuleNotFoundError as exc:
        raise CrewAIUnavailableError("CrewAI is required to create a DeepSeek LLM") from exc

    kwargs: dict[str, Any] = {
        "model": f"deepseek/{settings.deepseek_model}",
        "api_key": (
            settings.deepseek_api_key.get_secret_value()
            if settings.deepseek_api_key is not None
            else None
        ),
        "base_url": settings.deepseek_base_url,
        "temperature": settings.agent_temperature,
        "timeout": settings.agent_timeout_seconds,
        "max_retries": settings.agent_max_retries,
    }
    kwargs.update(overrides)

    try:
        return LLM(**kwargs)
    except TypeError:
        # Some CrewAI/LiteLLM versions expose the OpenAI-compatible endpoint as api_base.
        kwargs["api_base"] = kwargs.pop("base_url")
        return LLM(**kwargs)
