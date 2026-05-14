import json
from pathlib import Path

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
from app.services.conflict_resolution_service import resolve_conflicts


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def load_sample(name: str) -> dict:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def load_agent_outputs() -> AgentOutputBundle:
    return AgentOutputBundle.model_validate(load_sample("normal_final_decision.json")["agent_outputs"])


def test_high_risk_bullish_market_downgrades_buy_to_hold() -> None:
    outputs_payload = load_sample("normal_final_decision.json")["agent_outputs"]
    outputs_payload["risk_agent"]["risk_label"] = RiskLabel.HIGH
    outputs = AgentOutputBundle.model_validate(outputs_payload)

    assessment = resolve_conflicts(
        outputs,
        proposed_recommendation=Recommendation.BUY,
        time_horizon=InvestmentHorizon.SHORT_TERM,
    )

    assert assessment.recommendation == Recommendation.HOLD
    assert assessment.conflict_level == ConflictLevel.HIGH
    assert assessment.major_signal_conflict is True


def test_overvalued_high_risk_downgrades_increase_weight_to_maintain() -> None:
    outputs_payload = load_sample("normal_final_decision.json")["agent_outputs"]
    outputs_payload["risk_agent"]["risk_label"] = RiskLabel.HIGH
    outputs_payload["valuation_agent"]["valuation_label"] = ValuationLabel.OVERVALUED
    outputs = AgentOutputBundle.model_validate(outputs_payload)

    assessment = resolve_conflicts(
        outputs,
        proposed_portfolio_action=PortfolioAction.INCREASE_WEIGHT,
    )

    assert assessment.portfolio_action == PortfolioAction.MAINTAIN_WEIGHT
    assert any("overvalued" in signal.lower() for signal in assessment.conflicting_signals)


def test_short_term_bearish_sentiment_bullish_market_is_medium_conflict() -> None:
    outputs_payload = load_sample("normal_final_decision.json")["agent_outputs"]
    outputs_payload["sentiment_agent"]["sentiment_label"] = SentimentLabel.BEARISH
    outputs_payload["market_data_agent"]["market_signals"][0]["stance"] = SignalStance.BULLISH
    outputs = AgentOutputBundle.model_validate(outputs_payload)

    assessment = resolve_conflicts(
        outputs,
        proposed_recommendation=Recommendation.HOLD,
        time_horizon=InvestmentHorizon.SHORT_TERM,
    )

    assert assessment.recommendation == Recommendation.HOLD
    assert assessment.conflict_level == ConflictLevel.MEDIUM
