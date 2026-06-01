import json
from pathlib import Path

from app.schemas.enums import InvestmentHorizon, PortfolioAction, Recommendation
from app.schemas.manager_outputs import ManagerSynthesisOutput
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle
from app.application.services.critic_service import run_critic_debate_stage
from conftest import fixture_agent_outputs


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def load_sample(name: str) -> dict:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def test_critic_debate_can_downgrade_buy_under_bearish_pressure() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("high_risk_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("high_risk_tool_results.json"))
    outputs = fixture_agent_outputs(request, bundle)
    synthesis = ManagerSynthesisOutput(
        summary="Draft BUY before critic debate.",
        time_horizon=InvestmentHorizon.SHORT_TERM,
        proposed_recommendation=Recommendation.BUY,
        data_citations=["postgresql.real_time_prices:AAPL"],
    )

    revised = run_critic_debate_stage(synthesis=synthesis, agent_outputs=outputs)

    assert revised.debate_applied is True
    assert revised.proposed_recommendation == Recommendation.HOLD
    assert revised.debate_summary is not None
    assert revised.bearish_critic_points
    assert "CRITIC_DEBATE_APPLIED" in revised.limitations


def test_critic_debate_preserves_hold_when_signals_balanced() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("normal_tool_results.json"))
    outputs = fixture_agent_outputs(request, bundle)
    synthesis = ManagerSynthesisOutput(
        summary="Balanced HOLD draft.",
        time_horizon=InvestmentHorizon.SHORT_TERM,
        proposed_recommendation=Recommendation.HOLD,
        data_citations=["postgresql.real_time_prices:AAPL"],
    )

    revised = run_critic_debate_stage(synthesis=synthesis, agent_outputs=outputs)

    assert revised.proposed_recommendation in {Recommendation.HOLD, Recommendation.WATCH}
    assert revised.debate_applied is True


def test_critic_debate_can_downgrade_portfolio_increase_weight() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("high_risk_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("high_risk_tool_results.json"))
    outputs = fixture_agent_outputs(request, bundle)
    synthesis = ManagerSynthesisOutput(
        summary="Increase exposure draft.",
        time_horizon=InvestmentHorizon.SHORT_TERM,
        proposed_portfolio_action=PortfolioAction.INCREASE_WEIGHT,
        portfolio_allocation=[],
        portfolio_summary=None,
        proposed_recommendation=Recommendation.HOLD,
        data_citations=["postgresql.risk_features:AAPL"],
    )

    revised = run_critic_debate_stage(synthesis=synthesis, agent_outputs=outputs)

    assert revised.proposed_portfolio_action == PortfolioAction.MAINTAIN_WEIGHT
