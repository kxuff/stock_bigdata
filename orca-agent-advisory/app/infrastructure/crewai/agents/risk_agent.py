from typing import Any, Sequence

from app.infrastructure.crewai.config_loader import agent_config
from app.schemas.agent_outputs import RiskAgentOutput
from app.schemas.enums import AgentStatus, DecisionMode, RiskLabel, ToolStatus
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle

try:
    from crewai import Agent
except ModuleNotFoundError:
    Agent = None


def create_risk_agent(
    *,
    llm: Any,
    tools: Sequence[Any],
    verbose: bool = False,
) -> Any:
    _require_crewai()
    return Agent(
        config=agent_config("risk_agent"),
        llm=llm,
        tools=list(tools),
        verbose=verbose,
        allow_delegation=False,
    )


def _require_crewai() -> None:
    if Agent is None:
        raise RuntimeError("CrewAI is required to create Risk Agent")
