from typing import Any, Sequence

from app.infrastructure.crewai.config_loader import agent_config
from app.schemas.agent_outputs import ValuationAgentOutput
from app.schemas.enums import AgentStatus, ToolStatus, ValuationLabel
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle

try:
    from crewai import Agent
except ModuleNotFoundError:
    Agent = None


def create_valuation_agent(
    *,
    llm: Any,
    tools: Sequence[Any],
    verbose: bool = False,
) -> Any:
    _require_crewai()
    return Agent(
        config=agent_config("valuation_agent"),
        llm=llm,
        tools=list(tools),
        verbose=verbose,
        allow_delegation=False,
    )


def _require_crewai() -> None:
    if Agent is None:
        raise RuntimeError("CrewAI is required to create Valuation Agent")
