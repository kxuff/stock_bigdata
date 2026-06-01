from typing import Any

from app.infrastructure.crewai.config_loader import agent_config

try:
    from crewai import Agent
except ModuleNotFoundError:
    Agent = None


def create_manager_agent(
    *,
    llm: Any,
    verbose: bool = False,
    max_iter: int = 12,
    max_execution_time: int | None = None,
) -> Any:
    _require_crewai()
    config = agent_config("manager_agent")
    if max_execution_time is None:
        return Agent(
            config=config,
            llm=llm,
            allow_delegation=True,
            max_iter=max_iter,
            verbose=verbose,
        )
    return Agent(
        config=config,
        llm=llm,
        allow_delegation=True,
        max_iter=max_iter,
        max_execution_time=max_execution_time,
        verbose=verbose,
    )


def _require_crewai() -> None:
    if Agent is None:
        raise RuntimeError("CrewAI is required to create Manager Agent")
