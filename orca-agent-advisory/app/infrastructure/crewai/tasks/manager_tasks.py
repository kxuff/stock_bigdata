from typing import Any

from app.infrastructure.crewai.config_loader import crewai_task_config
from app.schemas.request import AdvisoryDecisionRequest

try:
    from crewai import Task
except ModuleNotFoundError:
    Task = None


def create_manager_synthesis_task(
    request: AdvisoryDecisionRequest,
    specialist_tasks: list[Any],
) -> Any:
    _require_crewai()
    return Task(
        config=crewai_task_config("manager_synthesis_task"),
        context=specialist_tasks,
    )


def _require_crewai() -> None:
    if Task is None:
        raise RuntimeError("CrewAI is required to create Manager Task")
