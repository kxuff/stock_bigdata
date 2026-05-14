from typing import Any

from app.crews.config_loader import agent_config

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
    kwargs: dict[str, Any] = agent_config("manager_agent")
    kwargs.update(
        {
            "llm": llm,
            "allow_delegation": True,
            "max_iter": max_iter,
            "verbose": verbose,
        }
    )
    if max_execution_time is not None:
        kwargs["max_execution_time"] = max_execution_time
    return Agent(**kwargs)


def _require_crewai() -> None:
    if Agent is None:
        raise RuntimeError("CrewAI is required to create Manager Agent")
