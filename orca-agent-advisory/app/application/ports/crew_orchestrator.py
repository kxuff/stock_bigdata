from dataclasses import dataclass
from typing import Any, Protocol

from app.schemas.agent_outputs import AgentOutputBundle
from app.schemas.manager_outputs import ManagerSynthesisOutput
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle


@dataclass
class CrewOrchestratedOutputs:
    agent_outputs: AgentOutputBundle
    manager_payload: Any | None


class CrewOrchestrator(Protocol):
    def run_orchestrated(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
    ) -> CrewOrchestratedOutputs:
        """Run crew orchestration and return specialist plus manager outputs."""

    def revise_manager_synthesis(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
        agent_outputs: AgentOutputBundle,
        previous_synthesis: ManagerSynthesisOutput,
        violations: list[str],
        attempt: int,
    ) -> Any:
        """Revise manager synthesis after deterministic validation failure."""
