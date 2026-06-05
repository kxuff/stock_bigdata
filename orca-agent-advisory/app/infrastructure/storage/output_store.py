import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from app.schemas.decision import PortfolioDecision, SingleSymbolDecision
from app.schemas.manager_outputs import ManagerSynthesisOutput
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle


DecisionResult: TypeAlias = SingleSymbolDecision | PortfolioDecision


@dataclass(frozen=True)
class DecisionOutputStore:
    output_dir: Path

    def save(
        self,
        *,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
        manager_synthesis: ManagerSynthesisOutput,
        decision: DecisionResult,
        revision_attempts: list[dict] | None = None,
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{decision.run_id}.json"
        payload = {
            "request": request.model_dump(mode="json"),
            "tool_results": tool_results.model_dump(mode="json", exclude_none=True),
            "manager_synthesis": manager_synthesis.model_dump(mode="json"),
            "final_decision": decision.model_dump(mode="json"),
            "revision_attempts": revision_attempts or [],
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return path
