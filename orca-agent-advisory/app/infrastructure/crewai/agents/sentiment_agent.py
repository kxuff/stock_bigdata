from typing import Any, Sequence

from crewai import Agent as CrewAgent

from app.infrastructure.crewai.config_loader import agent_config


def create_sentiment_agent(
    *,
    llm: Any,
    tools: Sequence[Any],
    verbose: bool = False,
) -> Any:
    return CrewAgent(
        config=agent_config("sentiment_agent"),
        llm=llm,
        tools=list(tools),
        verbose=verbose,
        allow_delegation=False,
        max_iter=4,
        max_execution_time=60,
    )
