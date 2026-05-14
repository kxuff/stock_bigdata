import json
from pathlib import Path

import pytest

from app.config import AgentSettings
from app.schemas.decision import (
    PortfolioAllocation,
    PortfolioDecision,
    PortfolioSummary,
    SingleSymbolDecision,
)
from app.schemas.enums import PortfolioAction, Recommendation, ReviewReason, RiskLabel
from app.schemas.manager_outputs import ManagerSynthesisOutput
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle, ToolResultValidationError
from app.services.decision_service import AdvisoryDecisionService, DecisionValidationError


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def load_sample(name: str) -> dict:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def decide(request_sample: str, tool_result_sample: str, output_dir: Path):
    request = AdvisoryDecisionRequest.model_validate(load_sample(request_sample))
    bundle = ToolResultBundle.model_validate(load_sample(tool_result_sample))
    service = AdvisoryDecisionService(
        settings=AgentSettings(
            advisory_use_crewai_manager=False,
            advisory_output_dir=output_dir,
        )
    )
    return service.decide(request, bundle)


def test_normal_single_symbol_flow_returns_final_decision_with_audit(tmp_path: Path) -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("normal_tool_results.json"))
    service = AdvisoryDecisionService(
        settings=AgentSettings(
            advisory_use_crewai_manager=False,
            advisory_output_dir=tmp_path,
        )
    )

    decision = service.decide(request, bundle)
    output_path = tmp_path / f"{decision.run_id}.json"

    assert isinstance(decision, SingleSymbolDecision)
    assert decision.request_id == "req_20260513_001"
    assert decision.not_financial_advice is True
    assert decision.confidence == decision.confidence_breakdown.risk_adjusted_confidence
    assert decision.audit.input_request_hash.startswith("sha256:")
    assert decision.retrieved_tool_audit.tool_result_bundle_hash.startswith("sha256:")
    assert decision.data_citations
    assert decision.agent_outputs.risk_agent.risk_label == RiskLabel.MEDIUM
    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["request"]["request_id"] == request.request_id
    assert saved["manager_synthesis"]["proposed_recommendation"] == decision.recommendation
    assert saved["final_decision"]["run_id"] == decision.run_id


def test_high_risk_conflict_caps_confidence_and_requires_review(tmp_path: Path) -> None:
    decision = decide("high_risk_request.json", "high_risk_tool_results.json", tmp_path)

    assert isinstance(decision, SingleSymbolDecision)
    assert decision.recommendation == Recommendation.HOLD
    assert decision.confidence <= 0.55
    assert decision.requires_human_review is True
    assert ReviewReason.HIGH_RISK in decision.review_reasons
    assert ReviewReason.CONFLICTING_SIGNALS in decision.review_reasons
    assert decision.conflicting_signals


def test_missing_optional_tool_result_returns_degraded_valid_response(tmp_path: Path) -> None:
    decision = decide("normal_request.json", "missing_sentiment_tool_results.json", tmp_path)

    assert isinstance(decision, SingleSymbolDecision)
    assert decision.agent_outputs.sentiment_agent is not None
    assert decision.agent_outputs.sentiment_agent.status == "SKIPPED"
    assert "SENTIMENT_CONTEXT_UNAVAILABLE" in decision.limitations
    assert decision.not_financial_advice is True


def test_portfolio_flow_returns_valid_allocation(tmp_path: Path) -> None:
    decision = decide(
        "portfolio_allocation_request.json",
        "portfolio_allocation_tool_results.json",
        tmp_path,
    )

    assert isinstance(decision, PortfolioDecision)
    assert sum(allocation.weight_pct for allocation in decision.portfolio_allocation) == pytest.approx(
        100.0
    )
    assert decision.validation_result.passed is True
    assert decision.not_financial_advice is True


def test_missing_required_market_tool_result_fails_before_agents() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("unavailable_market_tool_results.json"))
    service = AdvisoryDecisionService(settings=AgentSettings(advisory_use_crewai_manager=False))

    with pytest.raises(ToolResultValidationError, match="market_features is required"):
        service.decide(request, bundle)


def test_single_symbol_manager_synthesis_requires_recommendation(tmp_path: Path) -> None:
    class PortfolioDraftRunner:
        def run_manager_synthesis(
            self,
            request: AdvisoryDecisionRequest,
            tool_results: ToolResultBundle,
        ) -> ManagerSynthesisOutput:
            return ManagerSynthesisOutput(
                summary="Portfolio-style draft was returned for a single-symbol request.",
                time_horizon=request.user_context.investment_horizon,
                portfolio_allocation=[
                    PortfolioAllocation(
                        symbol=request.symbols[0],
                        weight_pct=100.0,
                        portfolio_action=PortfolioAction.MAINTAIN_WEIGHT,
                        rationale="Invalid draft mode for this request.",
                    )
                ],
                portfolio_summary=PortfolioSummary(
                    expected_risk_label=RiskLabel.MEDIUM,
                    concentration_risk=RiskLabel.MEDIUM,
                    dominant_themes=[],
                ),
            )

    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("normal_tool_results.json"))
    service = AdvisoryDecisionService(
        settings=AgentSettings(advisory_use_crewai_manager=True, advisory_output_dir=tmp_path),
        crew_runner=PortfolioDraftRunner(),
    )

    with pytest.raises(DecisionValidationError, match="proposed_recommendation"):
        service.decide(request, bundle)
