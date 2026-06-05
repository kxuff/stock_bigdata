from typing import Any, Sequence

from crewai import Agent as CrewAgent

from app.infrastructure.crewai.config_loader import agent_config


def create_valuation_agent(
    *,
    llm: Any,
    tools: Sequence[Any],
    verbose: bool = False,
) -> Any:
    return CrewAgent(
        config=agent_config("valuation_agent"),
        llm=llm,
        tools=list(tools),
        verbose=verbose,
        allow_delegation=False,
    )

