from typing import Any

from app.crews.config_loader import crewai_task_config

try:
    from crewai import Task
except ModuleNotFoundError:
    Task = None


def create_market_data_task(agent: Any) -> Any:
    _require_crewai()
    return Task(
        config=crewai_task_config("market_data_task"),
        agent=agent,
    )


def _require_crewai() -> None:
    if Task is None:
        raise RuntimeError("CrewAI is required to create Market Data Task")
