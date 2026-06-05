from typing import Any, cast

from app.infrastructure.crewai.config_loader import crewai_task_config

from crewai import Task

CrewTask = cast(Any, Task)


def create_valuation_task(agent: Any) -> Any:
    return CrewTask(
        config=crewai_task_config("valuation_task"),
        agent=agent,
    )

