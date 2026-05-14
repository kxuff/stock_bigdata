from app.schemas.enums import RiskLabel
from app.services.confidence_service import ConfidenceInputs, aggregate_confidence


def test_confidence_aggregates_weighted_inputs_and_caps_by_high_risk() -> None:
    breakdown = aggregate_confidence(
        ConfidenceInputs(
            market_confidence=0.9,
            ml_confidence=0.9,
            sentiment_confidence=0.8,
            valuation_confidence=0.8,
            source_quality_score=0.9,
            risk_label=RiskLabel.HIGH,
        )
    )

    assert breakdown.base_confidence == 0.87
    assert breakdown.risk_adjustment == -0.12
    assert breakdown.risk_cap == 0.65
    assert breakdown.risk_adjusted_confidence == 0.65


def test_confidence_caps_stale_data_at_55_percent() -> None:
    breakdown = aggregate_confidence(
        ConfidenceInputs(
            market_confidence=0.95,
            ml_confidence=0.95,
            sentiment_confidence=0.95,
            valuation_confidence=0.95,
            source_quality_score=0.8,
            source_quality_cap=0.9,
            risk_label=RiskLabel.LOW,
            data_freshness_score=0.4,
        )
    )

    assert breakdown.source_quality_cap == 0.55
    assert breakdown.risk_adjusted_confidence == 0.55


def test_confidence_caps_major_conflict_at_70_percent() -> None:
    breakdown = aggregate_confidence(
        ConfidenceInputs(
            market_confidence=0.95,
            ml_confidence=0.95,
            sentiment_confidence=0.95,
            valuation_confidence=0.95,
            source_quality_score=0.95,
            risk_label=RiskLabel.LOW,
            major_signal_conflict=True,
        )
    )

    assert breakdown.risk_cap == 0.70
    assert breakdown.risk_adjusted_confidence == 0.70
