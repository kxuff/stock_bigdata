from app.application.decision.decision_helpers import agent_confidence, portfolio_summary, tool_citations, unique
from app.config import AgentSettings
from app.schemas.agent_outputs import AgentOutputBundle
from app.schemas.decision import PortfolioDecision, ValidationResult
from app.schemas.enums import DecisionMode
from app.schemas.manager_outputs import ManagerSynthesisOutput
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle
from app.application.services.audit_service import create_audit_metadata, create_retrieved_tool_audit
from app.application.services.confidence_service import ConfidenceInputs, aggregate_confidence
from app.application.services.conflict_resolution_service import resolve_conflicts
from app.application.services.human_review_service import evaluate_human_review
from app.application.services.source_quality_service import SourceQualityAssessment
from app.validators.financial_validator import validate_financial_output


def assemble_portfolio_decision(
    *,
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
    agent_outputs: AgentOutputBundle,
    synthesis: ManagerSynthesisOutput,
    source_quality_assessment: SourceQualityAssessment,
    settings: AgentSettings,
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
        data_citations=unique([*synthesis.data_citations, *tool_citations(tool_results)]),
        debate_applied=synthesis.debate_applied,
        debate_summary=synthesis.debate_summary,
        bullish_critic_points=synthesis.bullish_critic_points,
        bearish_critic_points=synthesis.bearish_critic_points,
        not_financial_advice=True,
        risk_profile=request.user_context.risk_tolerance,
        portfolio_allocation=synthesis.portfolio_allocation,
        portfolio_summary=synthesis.portfolio_summary or portfolio_summary(agent_outputs),
        reasoning_trace=[
            synthesis.summary,
            "Deterministic confidence, conflict, validation, review, and audit policies applied.",
        ],
        validation_result=ValidationResult(passed=True, violations=[]),
    )
    validation = validate_financial_output(decision.model_dump(mode="json"), request=request)
    decision.validation_result = validation
    if not validation.passed:
        from app.application.decision.decision_helpers import DecisionValidationError

        raise DecisionValidationError("; ".join(validation.violations))
    return decision
