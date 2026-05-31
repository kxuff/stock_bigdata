from typing import Protocol

from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle


class ToolResultProvider(Protocol):
    def get_tool_results(self, request: AdvisoryDecisionRequest) -> ToolResultBundle:
        """Return upstream tool results for advisory request."""
