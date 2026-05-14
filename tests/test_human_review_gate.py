from app.schemas.enums import ConflictLevel, ReviewReason, RiskLabel
from app.services.human_review_service import evaluate_human_review


def test_human_review_not_required_for_clear_low_risk_case() -> None:
    decision = evaluate_human_review(
        final_confidence=0.74,
        risk_label=RiskLabel.MEDIUM,
        source_quality_score=0.9,
        stale_data=False,
        conflict_level=ConflictLevel.LOW,
    )

    assert decision.requires_human_review is False
    assert decision.review_reasons == []


def test_human_review_required_for_high_risk_low_confidence_and_stale_data() -> None:
    decision = evaluate_human_review(
        final_confidence=0.5,
        risk_label=RiskLabel.HIGH,
        source_quality_score=0.6,
        stale_data=True,
        conflict_level=ConflictLevel.HIGH,
    )

    assert decision.requires_human_review is True
    assert decision.review_reasons == [
        ReviewReason.LOW_CONFIDENCE,
        ReviewReason.HIGH_RISK,
        ReviewReason.DATA_QUALITY,
        ReviewReason.STALE_DATA,
        ReviewReason.CONFLICTING_SIGNALS,
    ]


def test_human_review_required_for_large_portfolio_turnover() -> None:
    decision = evaluate_human_review(
        final_confidence=0.8,
        risk_label=RiskLabel.LOW,
        portfolio_turnover_pct=30.0,
        portfolio_turnover_threshold_pct=25.0,
    )

    assert decision.requires_human_review is True
    assert decision.review_reasons == [ReviewReason.PORTFOLIO_CONSTRAINT_VIOLATION]
