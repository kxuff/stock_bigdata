from dataclasses import dataclass, field
from typing import Iterable

from pydantic import ValidationError

from app.agents.data_agent import analyze_market_data
from app.agents.risk_agent import analyze_risk
from app.agents.sentiment_agent import analyze_sentiment
from app.agents.valuation_agent import analyze_valuation
from app.config import AgentSettings, load_settings
from app.schemas.agent_outputs import AgentOutputBundle, BaseAgentOutput
from app.schemas.decision import (
    DecisionRationale,
    PortfolioAllocation,
    PortfolioDecision,
    PortfolioSummary,
    SingleSymbolDecision,
    ValidationResult,
)
from app.schemas.enums import (
    AgentStatus,
    DecisionMode,
    FactorWeight,
    InvestmentHorizon,
    PortfolioAction,
    Recommendation,
    RiskLabel,
    SentimentLabel,
    SignalStance,
    ValuationLabel,
)
from app.schemas.manager_outputs import ManagerSynthesisOutput
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle
from app.services.audit_service import create_audit_metadata, create_retrieved_tool_audit
from app.services.confidence_service import ConfidenceInputs, aggregate_confidence
from app.services.conflict_resolution_service import resolve_conflicts
from app.services.crew_runner import HierarchicalCrewRunner
from app.services.human_review_service import evaluate_human_review
from app.services.output_store import DecisionOutputStore
from app.services.source_quality_service import SourceQualityAssessment, assess_source_quality
from app.validators.financial_validator import validate_financial_output


DecisionResult = SingleSymbolDecision | PortfolioDecision


class DecisionValidationError(ValueError):
    """Raised when assembled advisory output fails deterministic validation."""


@dataclass
class AdvisoryDecisionService:
    settings: AgentSettings = field(default_factory=load_settings)
    crew_runner: HierarchicalCrewRunner | None = None
    output_store: DecisionOutputStore | None = None

    def decide(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
    ) -> DecisionResult:
        tool_results.validate_required_for(request)
        agent_outputs = run_specialist_analysis(request, tool_results)
        source_quality_assessment = assess_source_quality(request, tool_results)
        synthesis = self._manager_synthesis(request, tool_results, agent_outputs)

        if request.decision_mode == DecisionMode.PORTFOLIO_RECOMMENDATION:
            decision = self._assemble_portfolio_decision(
                request=request,
                tool_results=tool_results,
                agent_outputs=agent_outputs,
                synthesis=synthesis,
                source_quality_assessment=source_quality_assessment,
            )
            self._save_output(request, tool_results, synthesis, decision)
            return decision

        decision = self._assemble_single_symbol_decision(
            request=request,
            tool_results=tool_results,
            agent_outputs=agent_outputs,
            synthesis=synthesis,
            source_quality_assessment=source_quality_assessment,
        )
        self._save_output(request, tool_results, synthesis, decision)
        return decision

    def _manager_synthesis(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
        agent_outputs: AgentOutputBundle,
    ) -> ManagerSynthesisOutput:
        if self.settings.advisory_use_crewai_manager:
            runner = self.crew_runner or HierarchicalCrewRunner(settings=self.settings)
            return runner.run_manager_synthesis(request, tool_results)
        return build_deterministic_manager_synthesis(request, tool_results, agent_outputs)

    def _save_output(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
        synthesis: ManagerSynthesisOutput,
        decision: DecisionResult,
    ) -> None:
        store = self.output_store or DecisionOutputStore(self.settings.advisory_output_dir)
        store.save(
            request=request,
            tool_results=tool_results,
            manager_synthesis=synthesis,
            decision=decision,
        )

    def _assemble_single_symbol_decision(
        self,
        *,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
        agent_outputs: AgentOutputBundle,
        synthesis: ManagerSynthesisOutput,
        source_quality_assessment: SourceQualityAssessment,
    ) -> SingleSymbolDecision:
        symbol = request.symbols[0]
        proposed_recommendation = _require_single_symbol_recommendation(synthesis)
        conflict = resolve_conflicts(
            agent_outputs,
            proposed_recommendation=proposed_recommendation,
            time_horizon=request.user_context.investment_horizon,
            source_quality_score=source_quality_assessment.source_quality.overall_quality_score,
            stale_data=source_quality_assessment.stale_data,
        )
        recommendation = conflict.recommendation or proposed_recommendation
        confidence_breakdown = aggregate_confidence(
            ConfidenceInputs(
                market_confidence=agent_outputs.market_data_agent.confidence,
                ml_confidence=_ml_confidence(symbol, tool_results),
                sentiment_confidence=_agent_confidence(agent_outputs.sentiment_agent),
                valuation_confidence=_agent_confidence(agent_outputs.valuation_agent),
                source_quality_score=source_quality_assessment.source_quality.overall_quality_score,
                source_quality_cap=source_quality_assessment.source_quality_cap,
                risk_label=agent_outputs.risk_agent.risk_label,
                risk_cap=agent_outputs.risk_agent.confidence_cap,
                data_freshness_score=source_quality_assessment.source_quality.freshness_score,
                major_signal_conflict=conflict.major_signal_conflict,
            )
        )
        human_review = evaluate_human_review(
            final_confidence=confidence_breakdown.risk_adjusted_confidence,
            risk_label=agent_outputs.risk_agent.risk_label,
            source_quality_score=source_quality_assessment.source_quality.overall_quality_score,
            stale_data=source_quality_assessment.stale_data,
            conflict_level=conflict.conflict_level,
        )
        audit = create_audit_metadata(request, tool_results, settings=self.settings)
        retrieved_tool_audit = create_retrieved_tool_audit(tool_results)

        decision = SingleSymbolDecision(
            request_id=request.request_id,
            run_id=audit.run_id,
            decision_mode=DecisionMode.SINGLE_SYMBOL_ADVISORY,
            confidence=confidence_breakdown.risk_adjusted_confidence,
            confidence_breakdown=confidence_breakdown,
            requires_human_review=human_review.requires_human_review,
            review_reasons=human_review.review_reasons,
            audit=audit,
            retrieved_tool_audit=retrieved_tool_audit,
            data_citations=_unique([*synthesis.data_citations, *_tool_citations(tool_results)]),
            not_financial_advice=True,
            symbol=symbol,
            recommendation=recommendation,
            time_horizon=synthesis.time_horizon,
            summary=_single_symbol_summary(symbol, recommendation, agent_outputs, synthesis),
            agent_outputs=agent_outputs,
            decision_rationale=synthesis.decision_rationale,
            supporting_signals=synthesis.supporting_signals,
            conflicting_signals=_unique([*synthesis.conflicting_signals, *conflict.conflicting_signals]),
            conflict_level=conflict.conflict_level,
            risk_warnings=_unique([*synthesis.risk_warnings, *agent_outputs.risk_agent.risk_factors]),
            limitations=_collect_limitations(agent_outputs, synthesis, source_quality_assessment.quality_warnings),
            source_quality=source_quality_assessment.source_quality,
        )
        _validate_decision_output(decision, request)
        return decision

    def _assemble_portfolio_decision(
        self,
        *,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
        agent_outputs: AgentOutputBundle,
        synthesis: ManagerSynthesisOutput,
        source_quality_assessment: SourceQualityAssessment,
    ) -> PortfolioDecision:
        conflict = resolve_conflicts(
            agent_outputs,
            proposed_portfolio_action=synthesis.proposed_portfolio_action,
            time_horizon=request.user_context.investment_horizon,
            source_quality_score=source_quality_assessment.source_quality.overall_quality_score,
            stale_data=source_quality_assessment.stale_data,
        )
        confidence_breakdown = aggregate_confidence(
            ConfidenceInputs(
                market_confidence=agent_outputs.market_data_agent.confidence,
                ml_confidence=None,
                sentiment_confidence=_agent_confidence(agent_outputs.sentiment_agent),
                valuation_confidence=_agent_confidence(agent_outputs.valuation_agent),
                source_quality_score=source_quality_assessment.source_quality.overall_quality_score,
                source_quality_cap=source_quality_assessment.source_quality_cap,
                risk_label=agent_outputs.risk_agent.risk_label,
                risk_cap=agent_outputs.risk_agent.confidence_cap,
                data_freshness_score=source_quality_assessment.source_quality.freshness_score,
                major_signal_conflict=conflict.major_signal_conflict,
            )
        )
        human_review = evaluate_human_review(
            final_confidence=confidence_breakdown.risk_adjusted_confidence,
            risk_label=agent_outputs.risk_agent.risk_label,
            source_quality_score=source_quality_assessment.source_quality.overall_quality_score,
            stale_data=source_quality_assessment.stale_data,
            conflict_level=conflict.conflict_level,
        )
        audit = create_audit_metadata(request, tool_results, settings=self.settings)
        retrieved_tool_audit = create_retrieved_tool_audit(tool_results)

        decision = PortfolioDecision(
            request_id=request.request_id,
            run_id=audit.run_id,
            decision_mode=DecisionMode.PORTFOLIO_RECOMMENDATION,
            confidence=confidence_breakdown.risk_adjusted_confidence,
            confidence_breakdown=confidence_breakdown,
            requires_human_review=human_review.requires_human_review,
            review_reasons=human_review.review_reasons,
            audit=audit,
            retrieved_tool_audit=retrieved_tool_audit,
            data_citations=_unique([*synthesis.data_citations, *_tool_citations(tool_results)]),
            not_financial_advice=True,
            risk_profile=request.user_context.risk_tolerance,
            portfolio_allocation=synthesis.portfolio_allocation,
            portfolio_summary=synthesis.portfolio_summary or _portfolio_summary(agent_outputs),
            reasoning_trace=[
                synthesis.summary,
                "Deterministic confidence, conflict, validation, review, and audit policies applied.",
            ],
            validation_result=ValidationResult(passed=True, violations=[]),
        )
        validation = validate_financial_output(decision.model_dump(mode="json"), request=request)
        decision.validation_result = validation
        if not validation.passed:
            raise DecisionValidationError("; ".join(validation.violations))
        return decision


def run_specialist_analysis(
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
) -> AgentOutputBundle:
    return AgentOutputBundle(
        market_data_agent=analyze_market_data(request, tool_results),
        sentiment_agent=analyze_sentiment(request, tool_results),
        valuation_agent=analyze_valuation(request, tool_results),
        risk_agent=analyze_risk(request, tool_results),
    )


def build_deterministic_manager_synthesis(
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
    agent_outputs: AgentOutputBundle,
) -> ManagerSynthesisOutput:
    if request.decision_mode == DecisionMode.PORTFOLIO_RECOMMENDATION:
        return _portfolio_synthesis(request, tool_results, agent_outputs)
    return _single_symbol_synthesis(request, tool_results, agent_outputs)


def _single_symbol_synthesis(
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
    agent_outputs: AgentOutputBundle,
) -> ManagerSynthesisOutput:
    symbol = request.symbols[0]
    recommendation = _derive_recommendation(agent_outputs)
    market_stance = _dominant_market_stance(agent_outputs)
    valuation_label = _valuation_label(agent_outputs)
    sentiment_label = _sentiment_label(agent_outputs)
    risk_label = agent_outputs.risk_agent.risk_label

    rationale = [
        DecisionRationale(
            factor="market_signal",
            stance=market_stance or SignalStance.NEUTRAL,
            weight=FactorWeight.HIGH,
            explanation=agent_outputs.market_data_agent.summary,
        ),
        DecisionRationale(
            factor="risk",
            stance=_risk_stance(risk_label),
            weight=FactorWeight.HIGH,
            explanation=", ".join(agent_outputs.risk_agent.risk_factors)
            or agent_outputs.risk_agent.summary,
        ),
    ]
    if sentiment_label not in {None, SentimentLabel.UNAVAILABLE}:
        rationale.append(
            DecisionRationale(
                factor="sentiment",
                stance=_sentiment_stance(sentiment_label),
                weight=FactorWeight.MEDIUM,
                explanation=agent_outputs.sentiment_agent.summary
                if agent_outputs.sentiment_agent is not None
                else "Sentiment unavailable.",
            )
        )
    if valuation_label not in {None, ValuationLabel.UNKNOWN}:
        rationale.append(
            DecisionRationale(
                factor="valuation",
                stance=_valuation_stance(valuation_label),
                weight=FactorWeight.MEDIUM,
                explanation=agent_outputs.valuation_agent.summary
                if agent_outputs.valuation_agent is not None
                else "Valuation unavailable.",
            )
        )

    supporting_signals = _supporting_signals(agent_outputs)
    conflicting_signals = _draft_conflicts(agent_outputs)
    return ManagerSynthesisOutput(
        summary=(
            f"{symbol} draft recommendation is {recommendation.value} based on "
            "specialist market, sentiment, valuation, and risk evidence."
        ),
        time_horizon=request.user_context.investment_horizon,
        proposed_recommendation=recommendation,
        decision_rationale=rationale,
        supporting_signals=supporting_signals,
        conflicting_signals=conflicting_signals,
        risk_warnings=agent_outputs.risk_agent.risk_factors,
        limitations=_agent_limitations(agent_outputs),
        data_citations=_tool_citations(tool_results),
    )


def _portfolio_synthesis(
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
    agent_outputs: AgentOutputBundle,
) -> ManagerSynthesisOutput:
    min_cash = float(request.user_context.custom_constraints.get("min_cash_weight", 0.0))
    cash_weight = min_cash if request.user_context.allow_cash_position else 0.0
    remaining_weight = max(0.0, 100.0 - cash_weight)
    per_symbol = remaining_weight / len(request.symbols)
    capped_symbol_weight = min(per_symbol, request.user_context.max_single_asset_weight)
    allocations = [
        PortfolioAllocation(
            symbol=symbol,
            weight_pct=round(capped_symbol_weight, 2),
            portfolio_action=PortfolioAction.MAINTAIN_WEIGHT,
            rationale="Maintain exposure within max_single_asset_weight.",
        )
        for symbol in request.symbols
    ]
    allocated = sum(allocation.weight_pct for allocation in allocations)
    if request.user_context.allow_cash_position:
        allocations.append(
            PortfolioAllocation(
                symbol="CASH",
                weight_pct=round(100.0 - allocated, 2),
                portfolio_action=PortfolioAction.CASH_BUFFER,
                rationale="Cash buffer absorbs concentration and minimum cash constraints.",
            )
        )

    return ManagerSynthesisOutput(
        summary="Portfolio draft keeps concentration within user constraints.",
        time_horizon=request.user_context.investment_horizon,
        proposed_portfolio_action=PortfolioAction.MAINTAIN_WEIGHT,
        portfolio_allocation=allocations,
        portfolio_summary=_portfolio_summary(agent_outputs),
        risk_warnings=agent_outputs.risk_agent.risk_factors,
        limitations=_agent_limitations(agent_outputs),
        data_citations=_tool_citations(tool_results),
    )


def _derive_recommendation(agent_outputs: AgentOutputBundle) -> Recommendation:
    market_stance = _dominant_market_stance(agent_outputs)
    sentiment_label = _sentiment_label(agent_outputs)
    valuation_label = _valuation_label(agent_outputs)
    risk_label = agent_outputs.risk_agent.risk_label

    if risk_label == RiskLabel.CRITICAL:
        return Recommendation.SELL if market_stance == SignalStance.BEARISH else Recommendation.HOLD
    if market_stance == SignalStance.BULLISH and valuation_label != ValuationLabel.OVERVALUED:
        return Recommendation.BUY
    if market_stance == SignalStance.BEARISH or sentiment_label == SentimentLabel.BEARISH:
        return Recommendation.SELL
    if valuation_label == ValuationLabel.OVERVALUED:
        return Recommendation.HOLD
    return Recommendation.HOLD


def _single_symbol_summary(
    symbol: str,
    recommendation: Recommendation,
    agent_outputs: AgentOutputBundle,
    synthesis: ManagerSynthesisOutput,
) -> str:
    if synthesis.summary:
        return synthesis.summary.replace("draft recommendation", "final recommendation")
    risk_label = agent_outputs.risk_agent.risk_label.value
    return f"{symbol} is rated {recommendation.value} after applying risk and evidence policies ({risk_label} risk)."


def _validate_decision_output(decision: DecisionResult, request: AdvisoryDecisionRequest) -> None:
    try:
        type(decision).model_validate(decision.model_dump(mode="json"))
    except ValidationError as exc:
        raise DecisionValidationError(str(exc)) from exc

    validation = validate_financial_output(decision.model_dump(mode="json"), request=request)
    if not validation.passed:
        raise DecisionValidationError("; ".join(validation.violations))


def _require_single_symbol_recommendation(synthesis: ManagerSynthesisOutput) -> Recommendation:
    if synthesis.proposed_recommendation is None:
        raise DecisionValidationError(
            "single_symbol_advisory manager synthesis must include proposed_recommendation"
        )
    return synthesis.proposed_recommendation


def _agent_confidence(output: BaseAgentOutput | None) -> float | None:
    if output is None or output.status == AgentStatus.SKIPPED:
        return None
    return output.confidence


def _ml_confidence(symbol: str, tool_results: ToolResultBundle) -> float | None:
    if tool_results.ml_predictions is None:
        return None
    prediction = tool_results.ml_predictions.data.get(symbol)
    if prediction is None:
        return None
    return max(prediction.probability_up, prediction.probability_down)


def _dominant_market_stance(agent_outputs: AgentOutputBundle) -> SignalStance | None:
    if not agent_outputs.market_data_agent.market_signals:
        return None
    signal = max(agent_outputs.market_data_agent.market_signals, key=lambda item: item.confidence)
    return signal.stance


def _sentiment_label(agent_outputs: AgentOutputBundle) -> SentimentLabel | None:
    if agent_outputs.sentiment_agent is None:
        return None
    return agent_outputs.sentiment_agent.sentiment_label


def _valuation_label(agent_outputs: AgentOutputBundle) -> ValuationLabel | None:
    if agent_outputs.valuation_agent is None:
        return None
    return agent_outputs.valuation_agent.valuation_label


def _sentiment_stance(label: SentimentLabel) -> SignalStance:
    if label == SentimentLabel.BULLISH:
        return SignalStance.BULLISH
    if label == SentimentLabel.BEARISH:
        return SignalStance.BEARISH
    if label == SentimentLabel.MIXED:
        return SignalStance.MIXED
    return SignalStance.NEUTRAL


def _valuation_stance(label: ValuationLabel) -> SignalStance:
    if label == ValuationLabel.UNDERVALUED:
        return SignalStance.BULLISH
    if label == ValuationLabel.OVERVALUED:
        return SignalStance.BEARISH
    return SignalStance.NEUTRAL


def _risk_stance(label: RiskLabel) -> SignalStance:
    if label in {RiskLabel.HIGH, RiskLabel.CRITICAL}:
        return SignalStance.BEARISH
    if label == RiskLabel.LOW:
        return SignalStance.BULLISH
    return SignalStance.NEUTRAL


def _supporting_signals(agent_outputs: AgentOutputBundle) -> list[str]:
    signals: list[str] = []
    for market_signal in agent_outputs.market_data_agent.market_signals:
        signals.extend(market_signal.drivers)
    if agent_outputs.sentiment_agent is not None:
        signals.extend(agent_outputs.sentiment_agent.top_drivers)
    if agent_outputs.valuation_agent is not None:
        signals.extend(agent_outputs.valuation_agent.valuation_drivers)
    return _unique(signals)


def _draft_conflicts(agent_outputs: AgentOutputBundle) -> list[str]:
    conflicts: list[str] = []
    market_stance = _dominant_market_stance(agent_outputs)
    risk_label = agent_outputs.risk_agent.risk_label
    valuation_label = _valuation_label(agent_outputs)
    sentiment_label = _sentiment_label(agent_outputs)
    if market_stance == SignalStance.BULLISH and risk_label in {RiskLabel.HIGH, RiskLabel.CRITICAL}:
        conflicts.append("Bullish market signal conflicts with high risk.")
    if valuation_label == ValuationLabel.OVERVALUED and risk_label in {RiskLabel.HIGH, RiskLabel.CRITICAL}:
        conflicts.append("Overvaluation compounds high-risk exposure.")
    if market_stance == SignalStance.BULLISH and sentiment_label == SentimentLabel.BEARISH:
        conflicts.append("Bullish technical signal conflicts with bearish sentiment.")
    return conflicts


def _collect_limitations(
    agent_outputs: AgentOutputBundle,
    synthesis: ManagerSynthesisOutput,
    quality_warnings: Iterable[str],
) -> list[str]:
    return _unique([*synthesis.limitations, *_agent_limitations(agent_outputs), *quality_warnings])


def _agent_limitations(agent_outputs: AgentOutputBundle) -> list[str]:
    limitations: list[str] = []
    for output in [
        agent_outputs.market_data_agent,
        agent_outputs.sentiment_agent,
        agent_outputs.valuation_agent,
        agent_outputs.risk_agent,
    ]:
        if output is None:
            continue
        limitations.extend(output.limitations)
        limitations.extend(f"missing:{field}" for field in output.missing_fields)
    return _unique(limitations)


def _tool_citations(tool_results: ToolResultBundle) -> list[str]:
    citations: list[str] = []
    for result in [
        tool_results.market_features,
        tool_results.ml_predictions,
        tool_results.sentiment_snapshot,
        tool_results.valuation_snapshot,
        tool_results.risk_snapshot,
        tool_results.portfolio_snapshot,
    ]:
        if result is not None:
            citations.extend(result.source_refs)
    return _unique(citations)


def _portfolio_summary(agent_outputs: AgentOutputBundle) -> PortfolioSummary:
    return PortfolioSummary(
        expected_risk_label=agent_outputs.risk_agent.risk_label,
        concentration_risk=agent_outputs.risk_agent.risk_label,
        dominant_themes=_unique(agent_outputs.risk_agent.risk_factors[:3]),
    )


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
