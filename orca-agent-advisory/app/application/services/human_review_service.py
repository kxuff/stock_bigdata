from pydantic import BaseModel, ConfigDict, Field

from app.schemas.enums import ConflictLevel, ReviewReason, RiskLabel


class HumanReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requires_human_review: bool
    review_reasons: list[ReviewReason] = Field(default_factory=list)


def evaluate_human_review(
    *,
    final_confidence: float,
    risk_label: RiskLabel | None = None,
    source_quality_score: float = 1.0,
    stale_data: bool = False,
    conflict_level: ConflictLevel = ConflictLevel.NONE,
    portfolio_turnover_pct: float | None = None,
    portfolio_turnover_threshold_pct: float = 25.0,
) -> HumanReviewDecision:
    reasons: list[ReviewReason] = []

    if final_confidence < 0.55:
        reasons.append(ReviewReason.LOW_CONFIDENCE)
    if risk_label in {RiskLabel.HIGH, RiskLabel.CRITICAL}:
        reasons.append(ReviewReason.HIGH_RISK)
    if source_quality_score < 0.65:
        reasons.append(ReviewReason.DATA_QUALITY)
    if stale_data:
        reasons.append(ReviewReason.STALE_DATA)
    if conflict_level in {ConflictLevel.MEDIUM, ConflictLevel.HIGH}:
        reasons.append(ReviewReason.CONFLICTING_SIGNALS)
    if (
        portfolio_turnover_pct is not None
        and portfolio_turnover_pct > portfolio_turnover_threshold_pct
    ):
        reasons.append(ReviewReason.PORTFOLIO_CONSTRAINT_VIOLATION)

    deduped_reasons = list(dict.fromkeys(reasons))
    return HumanReviewDecision(
        requires_human_review=bool(deduped_reasons),
        review_reasons=deduped_reasons,
    )
