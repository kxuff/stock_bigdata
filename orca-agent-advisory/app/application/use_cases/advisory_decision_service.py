from dataclasses import dataclass, field

from app.application.decision.decision_helpers import DecisionValidationError
from app.application.decision.portfolio_assembler import assemble_portfolio_decision
from app.application.decision.single_symbol_assembler import assemble_single_symbol_decision
from app.application.ports.crew_orchestrator import CrewOrchestratedOutputs, CrewOrchestrator
from app.application.ports.output_store import DecisionOutputStore
from app.config import AgentSettings, load_settings
from app.schemas.agent_outputs import AgentOutputBundle
from app.schemas.decision import PortfolioDecision, SingleSymbolDecision
from app.schemas.enums import DecisionMode
from app.schemas.manager_outputs import ManagerSynthesisOutput
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle
from app.application.services.source_quality_service import assess_source_quality
from app.validators.manager_synthesis_parser import parse_manager_synthesis_output


DecisionResult = SingleSymbolDecision | PortfolioDecision


@dataclass
class AdvisoryDecisionService:
    settings: AgentSettings = field(default_factory=load_settings)
    crew_runner: CrewOrchestrator | None = None
    output_store: DecisionOutputStore | None = None

    def decide(self, request: AdvisoryDecisionRequest, tool_results: ToolResultBundle) -> DecisionResult:
        tool_results.validate_required_for(request)
        agent_outputs, synthesis = self._orchestrate(request, tool_results)
        source_quality_assessment = assess_source_quality(request, tool_results)

        if request.decision_mode == DecisionMode.PORTFOLIO_RECOMMENDATION:
            decision = assemble_portfolio_decision(
                request=request,
                tool_results=tool_results,
                agent_outputs=agent_outputs,
                synthesis=synthesis,
                source_quality_assessment=source_quality_assessment,
                settings=self.settings,
            )
        else:
            decision = assemble_single_symbol_decision(
                request=request,
                tool_results=tool_results,
                agent_outputs=agent_outputs,
                synthesis=synthesis,
                source_quality_assessment=source_quality_assessment,
                settings=self.settings,
            )

        self._save_output(request, tool_results, synthesis, decision)
        return decision

    def _orchestrate(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
    ) -> tuple[AgentOutputBundle, ManagerSynthesisOutput]:
        if self.crew_runner is None:
            raise RuntimeError("crew_runner dependency is required")
        crew_outputs = self.crew_runner.run_orchestrated(request, tool_results)
        synthesis = _parse_manager_payload(crew_outputs, request=request)
        return crew_outputs.agent_outputs, synthesis

    def _save_output(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
        synthesis: ManagerSynthesisOutput,
        decision: DecisionResult,
    ) -> None:
        if self.output_store is None:
            raise RuntimeError("output_store dependency is required")
        self.output_store.save(
            request=request,
            tool_results=tool_results,
            manager_synthesis=synthesis,
            decision=decision,
        )


def _parse_manager_payload(
    crew_outputs: CrewOrchestratedOutputs,
    *,
    request: AdvisoryDecisionRequest,
) -> ManagerSynthesisOutput:
    payload = crew_outputs.manager_payload
    if payload is None:
        raise DecisionValidationError("manager synthesis payload is required from agent runtime")

    pydantic_output = getattr(payload, "pydantic", None)
    if isinstance(pydantic_output, ManagerSynthesisOutput):
        return pydantic_output
    if isinstance(payload, ManagerSynthesisOutput):
        return payload

    try:
        if isinstance(payload, dict):
            return ManagerSynthesisOutput.model_validate(payload)
        return parse_manager_synthesis_output(payload, request)
    except Exception as exc:
        raise DecisionValidationError("manager synthesis payload could not be parsed") from exc


__all__ = [
    "AdvisoryDecisionService",
    "DecisionResult",
    "DecisionValidationError",
]
