from dataclasses import dataclass, field

from pydantic import ValidationError

from app.application.decision.decision_helpers import DecisionValidationError
from app.application.decision.portfolio_assembler import assemble_portfolio_decision
from app.application.decision.single_symbol_assembler import assemble_single_symbol_decision
from app.application.ports.crew_orchestrator import CrewOrchestratedOutputs, CrewOrchestrator
from app.application.ports.output_store import DecisionOutputStore
from app.application.services.critic_service import run_critic_debate_stage
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
        revision_attempts: list[dict] = []
        last_error: DecisionValidationError | None = None

        for attempt_index in range(self.settings.advisory_max_revision_attempts + 1):
            try:
                decision = self._assemble_decision(request, tool_results, agent_outputs, synthesis)
                self._save_output(request, tool_results, synthesis, decision, revision_attempts)
                return decision
            except (DecisionValidationError, ValidationError) as exc:
                last_error = DecisionValidationError(str(exc))
                if attempt_index >= self.settings.advisory_max_revision_attempts:
                    break
                if self.crew_runner is None or not hasattr(self.crew_runner, "revise_manager_synthesis"):
                    break
                violations = _violation_list(last_error)
                synthesis = self._revise_synthesis(
                    request=request,
                    tool_results=tool_results,
                    agent_outputs=agent_outputs,
                    previous_synthesis=synthesis,
                    violations=violations,
                    attempt=attempt_index + 1,
                )
                revision_attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "violations": violations,
                        "manager_synthesis": synthesis.model_dump(mode="json"),
                    }
                )

        attempts = self.settings.advisory_max_revision_attempts
        message = str(last_error) if last_error is not None else "decision validation failed"
        raise DecisionValidationError(f"{message} after {attempts} revision attempts")

    def _assemble_decision(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
        agent_outputs: AgentOutputBundle,
        synthesis: ManagerSynthesisOutput,
    ) -> DecisionResult:
        if self.settings.advisory_enable_critic_stage:
            synthesis = run_critic_debate_stage(synthesis=synthesis, agent_outputs=agent_outputs)
        source_quality_assessment = assess_source_quality(request, tool_results)

        if request.decision_mode == DecisionMode.PORTFOLIO_RECOMMENDATION:
            return assemble_portfolio_decision(
                request=request,
                tool_results=tool_results,
                agent_outputs=agent_outputs,
                synthesis=synthesis,
                source_quality_assessment=source_quality_assessment,
                settings=self.settings,
            )
        return assemble_single_symbol_decision(
            request=request,
            tool_results=tool_results,
            agent_outputs=agent_outputs,
            synthesis=synthesis,
            source_quality_assessment=source_quality_assessment,
            settings=self.settings,
        )

    def _orchestrate(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
    ) -> tuple[AgentOutputBundle, ManagerSynthesisOutput]:
        if self.crew_runner is None:
            raise RuntimeError("crew_runner dependency is required")
        try:
            crew_outputs = self.crew_runner.run_orchestrated(request, tool_results)
        except ValueError as exc:
            if str(exc).startswith("specialist_output_invalid:"):
                raise DecisionValidationError(str(exc)) from exc
            raise
        synthesis = _parse_manager_payload(crew_outputs, request=request)
        return crew_outputs.agent_outputs, synthesis

    def _revise_synthesis(
        self,
        *,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
        agent_outputs: AgentOutputBundle,
        previous_synthesis: ManagerSynthesisOutput,
        violations: list[str],
        attempt: int,
    ) -> ManagerSynthesisOutput:
        if self.crew_runner is None:
            raise RuntimeError("crew_runner dependency is required")
        try:
            payload = self.crew_runner.revise_manager_synthesis(
                request,
                tool_results,
                agent_outputs,
                previous_synthesis,
                violations,
                attempt,
            )
            return _parse_revision_payload(payload, request=request)
        except DecisionValidationError:
            raise
        except Exception as exc:
            raise DecisionValidationError(f"manager revision attempt {attempt} failed") from exc

    def _save_output(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
        synthesis: ManagerSynthesisOutput,
        decision: DecisionResult,
        revision_attempts: list[dict] | None = None,
    ) -> None:
        if self.output_store is None:
            raise RuntimeError("output_store dependency is required")
        self.output_store.save(
            request=request,
            tool_results=tool_results,
            manager_synthesis=synthesis,
            decision=decision,
            revision_attempts=revision_attempts,
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


def _parse_revision_payload(
    payload: object,
    *,
    request: AdvisoryDecisionRequest,
) -> ManagerSynthesisOutput:
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
        raise DecisionValidationError("manager revision payload could not be parsed") from exc


def _violation_list(exc: DecisionValidationError) -> list[str]:
    text = str(exc)
    return [part.strip() for part in text.split(";") if part.strip()] or [text]


__all__ = [
    "AdvisoryDecisionService",
    "DecisionResult",
    "DecisionValidationError",
]
