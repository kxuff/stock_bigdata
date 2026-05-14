import json
from pathlib import Path
from typing import Protocol

from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle, ToolResultValidationError


class ToolResultProvider(Protocol):
    def get_tool_results(self, request: AdvisoryDecisionRequest) -> ToolResultBundle:
        """Return upstream tool results for an advisory request."""


class SampleToolResultProvider:
    """Demo provider that mimics read-only upstream layer lookups from sample JSON."""

    REQUEST_SAMPLE_MAP = {
        "req_20260513_001": "normal_tool_results.json",
        "req_20260513_002": "high_risk_tool_results.json",
        "req_20260513_010": "portfolio_allocation_tool_results.json",
    }

    def __init__(self, sample_dir: Path | None = None) -> None:
        self.sample_dir = sample_dir or Path(__file__).resolve().parents[2] / "samples"

    def get_tool_results(self, request: AdvisoryDecisionRequest) -> ToolResultBundle:
        sample_name = request.metadata.get("tool_results_sample")
        if sample_name is not None:
            sample_name = str(sample_name)
            _validate_sample_name(sample_name)
        else:
            sample_name = self.REQUEST_SAMPLE_MAP.get(request.request_id)

        if not sample_name:
            raise ToolResultValidationError(
                f"no upstream tool result bundle is available for request_id={request.request_id}"
            )

        path = self.sample_dir / sample_name
        if not path.exists():
            raise ToolResultValidationError(f"tool result sample not found: {sample_name}")

        return ToolResultBundle.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _validate_sample_name(sample_name: str) -> None:
    if Path(sample_name).name != sample_name or not sample_name.endswith(".json"):
        raise ToolResultValidationError("tool_results_sample must be a sample JSON filename")
