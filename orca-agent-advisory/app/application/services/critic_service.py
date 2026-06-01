from dataclasses import dataclass

from app.schemas.agent_outputs import AgentOutputBundle
from app.schemas.enums import PortfolioAction, Recommendation, RiskLabel, SentimentLabel, SignalStance, ValuationLabel
from app.schemas.manager_outputs import ManagerSynthesisOutput


@dataclass(frozen=True)
class DebateAssessment:
    bullish_points: list[str]
    bearish_points: list[str]
    recommendation_override: Recommendation | None
    portfolio_action_override: PortfolioAction | None
    summary: str


def run_critic_debate_stage(
    synthesis: ManagerSynthesisOutput,
    agent_outputs: AgentOutputBundle,
) -> ManagerSynthesisOutput:
    assessment = _assess_debate(agent_outputs, synthesis)
    revised = synthesis.model_copy(deep=True)
    revised.debate_applied = True
    revised.debate_summary = assessment.summary
    revised.bullish_critic_points = assessment.bullish_points
    revised.bearish_critic_points = assessment.bearish_points
    revised.supporting_signals = _dedupe([*revised.supporting_signals, *assessment.bullish_points])
    revised.conflicting_signals = _dedupe([*revised.conflicting_signals, *assessment.bearish_points])
    revised.limitations = _dedupe([*revised.limitations, "CRITIC_DEBATE_APPLIED"])

    if assessment.recommendation_override is not None:
        revised.proposed_recommendation = assessment.recommendation_override
    if assessment.portfolio_action_override is not None:
        revised.proposed_portfolio_action = assessment.portfolio_action_override
    return revised


def _assess_debate(
    agent_outputs: AgentOutputBundle,
    synthesis: ManagerSynthesisOutput,
) -> DebateAssessment:
    bullish_points: list[str] = []
    bearish_points: list[str] = []
    bull_score = 0
    bear_score = 0

    market_stance = _dominant_market_stance(agent_outputs)
    if market_stance == SignalStance.BULLISH:
        bull_score += 2
        bullish_points.append("Bull critic: market/ML stance is bullish.")
    elif market_stance == SignalStance.BEARISH:
        bear_score += 1
        bearish_points.append("Bear critic: market/ML stance is bearish.")

    sentiment_label = (
        agent_outputs.sentiment_agent.sentiment_label
        if agent_outputs.sentiment_agent is not None
        else SentimentLabel.UNAVAILABLE
    )
    if sentiment_label == SentimentLabel.BULLISH:
        bull_score += 1
        bullish_points.append("Bull critic: sentiment is supportive.")
    elif sentiment_label == SentimentLabel.BEARISH:
        bear_score += 1
        bearish_points.append("Bear critic: sentiment is bearish.")

    valuation_label = (
        agent_outputs.valuation_agent.valuation_label
        if agent_outputs.valuation_agent is not None
        else ValuationLabel.UNKNOWN
    )
    if valuation_label == ValuationLabel.UNDERVALUED:
        bull_score += 1
        bullish_points.append("Bull critic: valuation appears undervalued.")
    elif valuation_label == ValuationLabel.OVERVALUED:
        bear_score += 1
        bearish_points.append("Bear critic: valuation appears overvalued.")

    risk_label = agent_outputs.risk_agent.risk_label
    if risk_label == RiskLabel.LOW:
        bull_score += 1
        bullish_points.append("Bull critic: risk profile is low.")
    elif risk_label in {RiskLabel.HIGH, RiskLabel.CRITICAL}:
        bear_score += 2
        bearish_points.append("Bear critic: risk profile is high/critical.")

    recommendation_override = _recommendation_override(
        synthesis.proposed_recommendation,
        bull_score=bull_score,
        bear_score=bear_score,
    )
    portfolio_action_override = _portfolio_action_override(
        synthesis.proposed_portfolio_action,
        bull_score=bull_score,
        bear_score=bear_score,
    )
    summary = (
        f"Critic debate completed (bull_score={bull_score}, bear_score={bear_score}). "
        "Deterministic post-debate safeguards remain authoritative."
    )
    return DebateAssessment(
        bullish_points=bullish_points,
        bearish_points=bearish_points,
        recommendation_override=recommendation_override,
        portfolio_action_override=portfolio_action_override,
        summary=summary,
    )


def _recommendation_override(
    recommendation: Recommendation | None,
    *,
    bull_score: int,
    bear_score: int,
) -> Recommendation | None:
    if recommendation is None:
        return None
    if recommendation == Recommendation.BUY and bear_score >= bull_score:
        return Recommendation.HOLD
    if recommendation == Recommendation.HOLD and (bear_score - bull_score) >= 2:
        return Recommendation.WATCH
    if recommendation == Recommendation.SELL and (bull_score - bear_score) >= 2:
        return Recommendation.HOLD
    return recommendation


def _portfolio_action_override(
    action: PortfolioAction | None,
    *,
    bull_score: int,
    bear_score: int,
) -> PortfolioAction | None:
    if action is None:
        return None
    if action == PortfolioAction.INCREASE_WEIGHT and bear_score >= bull_score:
        return PortfolioAction.MAINTAIN_WEIGHT
    if action == PortfolioAction.MAINTAIN_WEIGHT and (bear_score - bull_score) >= 2:
        return PortfolioAction.CASH_BUFFER
    if action == PortfolioAction.EXIT and (bull_score - bear_score) >= 2:
        return PortfolioAction.DECREASE_WEIGHT
    return action


def _dominant_market_stance(agent_outputs: AgentOutputBundle) -> SignalStance | None:
    signals = agent_outputs.market_data_agent.market_signals
    if not signals:
        return None
    strongest = max(signals, key=lambda signal: signal.confidence)
    return strongest.stance


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
