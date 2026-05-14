from decimal import Decimal, ROUND_HALF_UP

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.decision import ConfidenceBreakdown
from app.schemas.enums import RiskLabel


class ConfidenceInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_confidence: float = Field(ge=0.0, le=1.0)
    ml_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sentiment_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    valuation_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source_quality_score: float = Field(ge=0.0, le=1.0)
    source_quality_cap: float = Field(default=0.9, ge=0.0, le=1.0)
    risk_label: RiskLabel = RiskLabel.MEDIUM
    risk_cap: float | None = Field(default=None, ge=0.0, le=1.0)
    data_freshness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    major_signal_conflict: bool = False


def aggregate_confidence(inputs: ConfidenceInputs) -> ConfidenceBreakdown:
    ml_confidence = _optional_confidence(inputs.ml_confidence)
    sentiment_confidence = _optional_confidence(inputs.sentiment_confidence)
    valuation_confidence = _optional_confidence(inputs.valuation_confidence)

    base_confidence = _round_score(
        (
            0.30 * inputs.market_confidence
            + 0.25 * ml_confidence
            + 0.20 * sentiment_confidence
            + 0.15 * valuation_confidence
            + 0.10 * inputs.source_quality_score
        ),
    )
    risk_adjustment = _risk_adjustment(inputs.risk_label)
    risk_adjusted_confidence = _round_score(
        max(0.0, min(1.0, base_confidence + risk_adjustment)),
    )

    risk_cap = inputs.risk_cap if inputs.risk_cap is not None else _risk_cap(inputs.risk_label)
    source_quality_cap = inputs.source_quality_cap
    if inputs.data_freshness_score < 0.6:
        source_quality_cap = min(source_quality_cap, 0.55)
    if inputs.major_signal_conflict:
        risk_cap = min(risk_cap, 0.70)

    final_confidence = _round_score(
        min(risk_adjusted_confidence, risk_cap, source_quality_cap),
    )

    return ConfidenceBreakdown(
        base_confidence=base_confidence,
        risk_adjusted_confidence=final_confidence,
        risk_cap=risk_cap,
        source_quality_cap=source_quality_cap,
        market_confidence=inputs.market_confidence,
        ml_confidence=inputs.ml_confidence,
        sentiment_confidence=inputs.sentiment_confidence,
        valuation_confidence=inputs.valuation_confidence,
        risk_adjustment=risk_adjustment,
        source_quality_adjustment=_round_score(
            min(0.0, (inputs.source_quality_score - 1.0) * 0.2),
        ),
    )


def _optional_confidence(value: float | None) -> float:
    return 0.5 if value is None else value


def _risk_adjustment(risk_label: RiskLabel) -> float:
    return {
        RiskLabel.LOW: 0.02,
        RiskLabel.MEDIUM: -0.04,
        RiskLabel.HIGH: -0.12,
        RiskLabel.CRITICAL: -0.25,
    }[risk_label]


def _risk_cap(risk_label: RiskLabel) -> float:
    return {
        RiskLabel.LOW: 0.95,
        RiskLabel.MEDIUM: 0.85,
        RiskLabel.HIGH: 0.65,
        RiskLabel.CRITICAL: 0.45,
    }[risk_label]


def _round_score(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
