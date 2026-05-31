from typing import Any

from app.config import AgentSettings, load_settings
from app.infrastructure.llm.deepseek_client import CrewAIUnavailableError


def create_llm(
    settings: AgentSettings | None = None,
    *,
    provider: str | None = None,
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
        raise CrewAIUnavailableError("CrewAI is required to create an LLM") from exc

    resolved_settings = settings or load_settings()
    resolved_provider = (provider or resolved_settings.llm_provider).strip().lower()
    resolved_model = _resolve_model(resolved_settings, resolved_provider, model)
    resolved_api_key = _resolve_api_key(resolved_settings, resolved_provider, api_key)
    resolved_base_url = base_url if base_url is not None else resolved_settings.llm_base_url
    resolved_temperature = resolved_settings.agent_temperature if temperature is None else temperature
    resolved_timeout = resolved_settings.agent_timeout_seconds if timeout is None else timeout
    resolved_max_retries = resolved_settings.agent_max_retries if max_retries is None else max_retries

    kwargs = {
        "model": resolved_model,
        "api_key": resolved_api_key,
        "temperature": resolved_temperature,
        "timeout": resolved_timeout,
        "max_retries": resolved_max_retries,
    }
    if resolved_base_url is not None:
        kwargs["base_url"] = resolved_base_url

    try:
        return LLM(**kwargs)
    except TypeError:
        if "base_url" in kwargs:
            kwargs["api_base"] = kwargs.pop("base_url")
        return LLM(**kwargs)


def create_deepseek_llm(
    settings: AgentSettings | None = None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
) -> Any:
    return create_llm(
        settings,
        provider="deepseek",
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )


def _resolve_model(settings: AgentSettings, provider: str, override: str | None) -> str:
    raw_model = override or settings.llm_model
    if provider == "deepseek":
        raw_model = raw_model or settings.deepseek_model
        return raw_model if raw_model.startswith("deepseek/") else f"deepseek/{raw_model}"
    if provider == "openai":
        raw_model = raw_model or "gpt-4o-mini"
        return raw_model if "/" in raw_model or raw_model.startswith("openai/") else f"openai/{raw_model}"
    if provider == "openai_compatible":
        if raw_model is None:
            raise ValueError(f"llm_model is required for provider {provider}")
        return raw_model if raw_model.startswith("openai/") else f"openai/{raw_model}"
    if provider == "litellm":
        if raw_model is None:
            raise ValueError(f"llm_model is required for provider {provider}")
        return raw_model
    raise ValueError("llm_provider must be one of: deepseek, openai, openai_compatible, litellm")


def _resolve_api_key(settings: AgentSettings, provider: str, override: str | None) -> str | None:
    if override is not None:
        return override
    secret = settings.llm_api_key
    if provider == "deepseek" and secret is None:
        secret = settings.deepseek_api_key
    return secret.get_secret_value() if secret is not None else None
