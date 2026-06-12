from typing import Any, cast

from app.infrastructure.crewai.config_loader import crewai_task_config
from app.schemas.manager_outputs import ManagerSynthesisOutput

from crewai import Task

CrewTask = cast(Any, Task)


def create_manager_synthesis_task(
    specialist_tasks: list[Any],
) -> Any:
    return CrewTask(
        config=crewai_task_config("manager_synthesis_task"),
        context=specialist_tasks,
        output_pydantic=ManagerSynthesisOutput,
    )
