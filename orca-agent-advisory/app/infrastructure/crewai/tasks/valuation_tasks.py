from typing import Any

from app.infrastructure.crewai.config_loader import crewai_task_config

try:
    from crewai import Task
except ModuleNotFoundError:
    Task = None


def create_valuation_task(agent: Any) -> Any:
    _require_crewai()
    return Task(
        config=crewai_task_config("valuation_task"),
        agent=agent,
    )


def _require_crewai() -> None:
    if Task is None:
        raise RuntimeError("CrewAI is required to create Valuation Task")
