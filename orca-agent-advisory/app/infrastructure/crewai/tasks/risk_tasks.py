from typing import Any, cast

from app.infrastructure.crewai.config_loader import crewai_task_config
from app.schemas.agent_outputs import RiskAgentOutput

from crewai import Task

CrewTask = cast(Any, Task)


def create_risk_task(agent: Any) -> Any:
    return CrewTask(
        config=crewai_task_config("risk_task"),
        agent=agent,
        output_pydantic=RiskAgentOutput,
    )
