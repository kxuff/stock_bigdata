from typing import Any

from app.config import AgentSettings, load_settings
from app.llm.deepseek_client import build_deepseek_llm


def create_deepseek_llm(settings: AgentSettings | None = None, **overrides: Any) -> Any:
    resolved_settings = settings or load_settings()
    return build_deepseek_llm(resolved_settings, **overrides)
