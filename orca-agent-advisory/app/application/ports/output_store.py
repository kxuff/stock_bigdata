from pathlib import Path
from typing import Protocol, TypeAlias

from app.schemas.decision import PortfolioDecision, SingleSymbolDecision
from app.schemas.manager_outputs import ManagerSynthesisOutput
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle


DecisionResult: TypeAlias = SingleSymbolDecision | PortfolioDecision


class DecisionOutputStore(Protocol):
    def save(
        self,
        *,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
        manager_synthesis: ManagerSynthesisOutput,
        decision: DecisionResult,
        revision_attempts: list[dict] | None = None,
    ) -> Path:
        """Persist decision payload and return path."""
