from typing import Any

from crewai import Agent as CrewAgent

from app.infrastructure.crewai.config_loader import agent_config


def create_manager_agent(
    *,
    llm: Any,
    verbose: bool = False,
    max_iter: int = 12,
    max_execution_time: int | None = None,
) -> Any:
    config = agent_config("manager_agent")
    if max_execution_time is None:
        return CrewAgent(
            config=config,
            llm=llm,
            allow_delegation=True,
            max_iter=max_iter,
            verbose=verbose,
        )
    return CrewAgent(
        config=config,
        llm=llm,
        allow_delegation=True,
        max_iter=max_iter,
        max_execution_time=max_execution_time,
        verbose=verbose,
    )
