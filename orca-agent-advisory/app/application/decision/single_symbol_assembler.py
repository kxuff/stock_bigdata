from app.application.decision.decision_helpers import (
    agent_confidence,
    collect_limitations,
    ml_confidence,
    tool_citations,
    unique,
    validate_decision_output,
)
from app.config import AgentSettings
from app.schemas.agent_outputs import AgentOutputBundle
from app.schemas.decision import SingleSymbolDecision
from app.schemas.enums import DecisionMode, Recommendation
from app.schemas.manager_outputs import ManagerSynthesisOutput
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle
from app.application.services.audit_service import create_audit_metadata, create_retrieved_tool_audit
from app.application.services.confidence_service import ConfidenceInputs, aggregate_confidence
from app.application.services.conflict_resolution_service import resolve_conflicts
from app.application.services.human_review_service import evaluate_human_review
from app.application.services.source_quality_service import SourceQualityAssessment


def assemble_single_symbol_decision(
    *,
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
    agent_outputs: AgentOutputBundle,
    synthesis: ManagerSynthesisOutput,
    source_quality_assessment: SourceQualityAssessment,
    settings: AgentSettings,
) -> SingleSymbolDecision:
    symbol = request.symbols[0]
    proposed_recommendation = require_single_symbol_recommendation(synthesis)
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
            ml_confidence=ml_confidence(symbol, tool_results),
            sentiment_confidence=agent_confidence(agent_outputs.sentiment_agent),
            valuation_confidence=agent_confidence(agent_outputs.valuation_agent),
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
    audit = create_audit_metadata(request, tool_results, settings=settings)
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
        data_citations=unique([*synthesis.data_citations, *tool_citations(tool_results)]),
        debate_applied=synthesis.debate_applied,
        debate_summary=synthesis.debate_summary,
        bullish_critic_points=synthesis.bullish_critic_points,
        bearish_critic_points=synthesis.bearish_critic_points,
        not_financial_advice=True,
        symbol=symbol,
        recommendation=recommendation,
        time_horizon=synthesis.time_horizon,
        summary=single_symbol_summary(symbol, recommendation, agent_outputs, synthesis),
        agent_outputs=agent_outputs,
        decision_rationale=synthesis.decision_rationale,
        supporting_signals=synthesis.supporting_signals,
        conflicting_signals=unique([*synthesis.conflicting_signals, *conflict.conflicting_signals]),
        conflict_level=conflict.conflict_level,
        risk_warnings=unique([*synthesis.risk_warnings, *agent_outputs.risk_agent.risk_factors]),
        limitations=collect_limitations(agent_outputs, synthesis, source_quality_assessment.quality_warnings),
        source_quality=source_quality_assessment.source_quality,
    )
    validate_decision_output(decision, request)
    return decision


def require_single_symbol_recommendation(synthesis: ManagerSynthesisOutput) -> Recommendation:
    if synthesis.proposed_recommendation is None:
        from app.application.decision.decision_helpers import DecisionValidationError

        raise DecisionValidationError(
            "single_symbol_advisory manager synthesis must include proposed_recommendation"
        )
    return synthesis.proposed_recommendation


def single_symbol_summary(
    symbol: str,
    recommendation: Recommendation,
    agent_outputs: AgentOutputBundle,
    synthesis: ManagerSynthesisOutput,
) -> str:
    if synthesis.summary:
        return synthesis.summary.replace("draft recommendation", "final recommendation")
    risk_label = agent_outputs.risk_agent.risk_label.value
    return f"{symbol} is rated {recommendation.value} after applying risk and evidence policies ({risk_label} risk)."
