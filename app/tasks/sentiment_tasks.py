from typing import Any

from app.crews.config_loader import crewai_task_config

try:
    from crewai import Task
except ModuleNotFoundError:
    Task = None


def create_sentiment_task(agent: Any) -> Any:
    _require_crewai()
    return Task(
        config=crewai_task_config("sentiment_task"),
        agent=agent,
    )


def _require_crewai() -> None:
    if Task is None:
        raise RuntimeError("CrewAI is required to create Sentiment Task")
