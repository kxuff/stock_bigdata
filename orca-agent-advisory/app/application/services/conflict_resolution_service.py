from pydantic import BaseModel, ConfigDict, Field

from app.schemas.agent_outputs import AgentOutputBundle
from app.schemas.enums import (
    ConflictLevel,
    InvestmentHorizon,
    PortfolioAction,
    Recommendation,
    RiskLabel,
    SentimentLabel,
    SignalStance,
    ValuationLabel,
)


class ConflictAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conflict_level: ConflictLevel
    major_signal_conflict: bool
    conflicting_signals: list[str] = Field(default_factory=list)
    recommendation: Recommendation | None = None
    portfolio_action: PortfolioAction | None = None


def resolve_conflicts(
    agent_outputs: AgentOutputBundle,
    *,
    proposed_recommendation: Recommendation | None = None,
    proposed_portfolio_action: PortfolioAction | None = None,
    time_horizon: InvestmentHorizon | None = None,
    source_quality_score: float = 1.0,
    stale_data: bool = False,
) -> ConflictAssessment:
    conflicts: list[str] = []
    risk_label = agent_outputs.risk_agent.risk_label
    market_stance = _dominant_market_stance(agent_outputs)
    sentiment_label = (
        agent_outputs.sentiment_agent.sentiment_label
        if agent_outputs.sentiment_agent is not None
        else None
    )
    valuation_label = (
        agent_outputs.valuation_agent.valuation_label
        if agent_outputs.valuation_agent is not None
        else None
    )

    if risk_label in {RiskLabel.HIGH, RiskLabel.CRITICAL} and market_stance == SignalStance.BULLISH:
        conflicts.append("Risk is high while market or ML signals are bullish.")
    if risk_label in {RiskLabel.HIGH, RiskLabel.CRITICAL} and valuation_label == ValuationLabel.OVERVALUED:
        conflicts.append("Valuation is overvalued while risk is high.")
    if sentiment_label == SentimentLabel.BEARISH and market_stance == SignalStance.BULLISH:
        conflicts.append("Sentiment is bearish while technical signals are bullish.")
    if source_quality_score < 0.7:
        conflicts.append("Source quality is low enough to cap confidence.")
    if stale_data:
        conflicts.append("Market or supporting data is stale.")

    recommendation = downgrade_recommendation(
        proposed_recommendation,
        risk_label=risk_label,
        source_quality_score=source_quality_score,
        stale_data=stale_data,
    )
    portfolio_action = downgrade_portfolio_action(
        proposed_portfolio_action,
        risk_label=risk_label,
        valuation_label=valuation_label,
    )

    conflict_level = _conflict_level(conflicts, risk_label, time_horizon)
    return ConflictAssessment(
        conflict_level=conflict_level,
        major_signal_conflict=conflict_level in {ConflictLevel.HIGH, ConflictLevel.MEDIUM},
        conflicting_signals=conflicts,
        recommendation=recommendation,
        portfolio_action=portfolio_action,
    )


def downgrade_recommendation(
    recommendation: Recommendation | None,
    *,
    risk_label: RiskLabel,
    source_quality_score: float,
    stale_data: bool,
) -> Recommendation | None:
    if recommendation != Recommendation.BUY:
        return recommendation
    if risk_label in {RiskLabel.HIGH, RiskLabel.CRITICAL}:
        return Recommendation.HOLD
    if source_quality_score < 0.6 or stale_data:
        return Recommendation.WATCH
    return recommendation


def downgrade_portfolio_action(
    portfolio_action: PortfolioAction | None,
    *,
    risk_label: RiskLabel,
    valuation_label: ValuationLabel | None,
) -> PortfolioAction | None:
    if portfolio_action != PortfolioAction.INCREASE_WEIGHT:
        return portfolio_action
    if risk_label in {RiskLabel.HIGH, RiskLabel.CRITICAL}:
        return PortfolioAction.MAINTAIN_WEIGHT
    if valuation_label == ValuationLabel.OVERVALUED:
        return PortfolioAction.MAINTAIN_WEIGHT
    return portfolio_action


def _dominant_market_stance(agent_outputs: AgentOutputBundle) -> SignalStance | None:
    signals = agent_outputs.market_data_agent.market_signals
    if not signals:
        return None
    strongest = max(signals, key=lambda signal: signal.confidence)
    return strongest.stance


def _conflict_level(
    conflicts: list[str],
    risk_label: RiskLabel,
    time_horizon: InvestmentHorizon | None,
) -> ConflictLevel:
    if not conflicts:
        return ConflictLevel.NONE
    if risk_label == RiskLabel.CRITICAL:
        return ConflictLevel.HIGH
    if len(conflicts) >= 2 or risk_label == RiskLabel.HIGH:
        return ConflictLevel.HIGH
    if time_horizon in {InvestmentHorizon.INTRADAY, InvestmentHorizon.SHORT_TERM}:
        return ConflictLevel.MEDIUM
    return ConflictLevel.LOW
