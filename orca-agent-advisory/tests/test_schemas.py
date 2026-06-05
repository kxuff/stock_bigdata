import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.decision import PortfolioDecision, SingleSymbolDecision
from app.schemas.agent import AgentQueryRequest
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle, ToolResultValidationError


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def load_sample(name: str) -> dict:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "sample_name",
    [
        "normal_request.json",
        "high_risk_request.json",
        "portfolio_allocation_request.json",
    ],
)
def test_request_samples_parse(sample_name: str) -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample(sample_name))

    assert request.request_id
    assert request.symbols


@pytest.mark.parametrize(
    "sample_name",
    [
        "normal_tool_results.json",
        "high_risk_tool_results.json",
        "missing_sentiment_tool_results.json",
        "missing_valuation_tool_results.json",
        "stale_data_tool_results.json",
        "unavailable_market_tool_results.json",
        "portfolio_allocation_tool_results.json",
    ],
)
def test_tool_result_samples_parse(sample_name: str) -> None:
    bundle = ToolResultBundle.model_validate(load_sample(sample_name))

    assert bundle.model_dump(exclude_none=True)


def test_final_single_symbol_decision_sample_parses() -> None:
    decision = SingleSymbolDecision.model_validate(load_sample("normal_final_decision.json"))

    assert decision.recommendation == "HOLD"
    assert decision.not_financial_advice is True
    assert decision.confidence_breakdown.risk_adjusted_confidence == pytest.approx(0.68)


def test_portfolio_decision_sample_parses_and_totals_100_percent() -> None:
    decision = PortfolioDecision.model_validate(load_sample("portfolio_allocation_final_decision.json"))

    assert sum(allocation.weight_pct for allocation in decision.portfolio_allocation) == pytest.approx(100.0)
    assert decision.validation_result.passed is True


def test_missing_required_request_fields_are_rejected() -> None:
    payload = load_sample("normal_request.json")
    payload.pop("request_id")

    with pytest.raises(ValidationError, match="request_id"):
        AdvisoryDecisionRequest.model_validate(payload)


def test_agent_query_request_accepts_conversation_history() -> None:
    request = AgentQueryRequest.model_validate(
        {
            "message": "what about it?",
            "conversation_id": "conv-1",
            "history": [{"role": "user", "content": "Analyze AAPL", "metadata": {"symbol": "AAPL"}, "created_at": "2026-01-02T03:04:05Z"}],
        }
    )

    assert request.conversation_id == "conv-1"
    assert request.history[0].role == "user"
    assert request.history[0].metadata == {"symbol": "AAPL"}


def test_agent_query_history_message_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        AgentQueryRequest.model_validate({"message": "hi", "history": [{"role": "user", "content": "Analyze AAPL", "unexpected": True}]})


def test_single_symbol_request_rejects_multiple_symbols() -> None:
    payload = load_sample("normal_request.json")
    payload["symbols"] = ["AAPL", "MSFT"]

    with pytest.raises(ValidationError, match="single_symbol_advisory requires exactly one symbol"):
        AdvisoryDecisionRequest.model_validate(payload)


def test_unavailable_required_market_tool_result_is_rejected() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("unavailable_market_tool_results.json"))

    with pytest.raises(ToolResultValidationError, match="market_features is required"):
        bundle.validate_required_for(request)


def test_stale_required_market_tool_result_is_rejected() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("stale_data_tool_results.json"))

    with pytest.raises(ToolResultValidationError, match="market_features is stale"):
        bundle.validate_required_for(request)


@pytest.mark.parametrize(
    "sample_name",
    [
        "missing_sentiment_tool_results.json",
        "missing_valuation_tool_results.json",
    ],
)
def test_missing_optional_tool_results_do_not_fail_request(sample_name: str) -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample(sample_name))

    bundle.validate_required_for(request)


def test_optional_sentiment_and_valuation_metadata_parse() -> None:
    payload = load_sample("normal_tool_results.json")
    payload["sentiment_snapshot"]["data"]["AAPL"].update(
        {
            "latest_article_published_at": "2026-05-13T14:00:00Z",
            "oldest_article_published_at": "2026-05-10T14:00:00Z",
            "sentiment_scored_at": "2026-05-13T15:00:00Z",
            "stale_article_count": 1,
        }
    )
    payload["valuation_snapshot"]["data"]["AAPL"].update(
        {
            "valuation_method": "relative_pe",
            "valuation_quality": "HIGH",
            "valuation_fetched_at": "2026-05-13T15:00:00Z",
            "fundamentals_as_of": "2026-03-31T00:00:00Z",
            "sector_sample_count": 32,
        }
    )

    bundle = ToolResultBundle.model_validate(payload)

    assert bundle.sentiment_snapshot.data["AAPL"].stale_article_count == 1
    assert bundle.valuation_snapshot.data["AAPL"].valuation_method == "relative_pe"


def test_portfolio_mode_requires_risk_and_portfolio_tool_results() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("portfolio_allocation_request.json"))
    payload = load_sample("portfolio_allocation_tool_results.json")
    payload.pop("portfolio_snapshot")
    incomplete_bundle = ToolResultBundle.model_validate(payload)

    with pytest.raises(ToolResultValidationError, match="portfolio_snapshot is required"):
        incomplete_bundle.validate_required_for(request)


def test_portfolio_tool_results_validate_for_portfolio_request() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("portfolio_allocation_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("portfolio_allocation_tool_results.json"))

    bundle.validate_required_for(request)


def test_invalid_portfolio_allocation_total_is_rejected() -> None:
    payload = load_sample("portfolio_allocation_final_decision.json")
    payload["portfolio_allocation"][0]["weight_pct"] = 34

    with pytest.raises(ValidationError, match="must total 100"):
        PortfolioDecision.model_validate(payload)


def test_final_output_contract_snapshot() -> None:
    decision = SingleSymbolDecision.model_validate(load_sample("normal_final_decision.json"))

    assert decision.model_dump(mode="json", exclude={"agent_outputs"}) == {
        "request_id": "req_20260513_001",
        "run_id": "run_20260513_001",
        "decision_mode": "single_symbol_advisory",
        "confidence": 0.68,
        "confidence_breakdown": {
            "base_confidence": 0.74,
            "risk_adjusted_confidence": 0.68,
            "risk_cap": 0.78,
            "source_quality_cap": 0.9,
            "market_confidence": 0.72,
            "ml_confidence": 0.68,
            "sentiment_confidence": 0.61,
            "valuation_confidence": 0.58,
            "risk_adjustment": -0.08,
            "source_quality_adjustment": -0.03,
        },
        "requires_human_review": False,
        "review_reasons": [],
        "audit": {
            "run_id": "run_20260513_001",
            "request_id": "req_20260513_001",
            "model_provider": "DeepSeek",
            "model_name": "deepseek-v4-flash",
            "framework": "CrewAI",
            "temperature": 0.2,
            "input_request_hash": "sha256:req001",
            "tool_result_bundle_hash": "sha256:tools001",
            "validator_version": "v1.0.0",
            "created_at": "2026-05-13T15:46:00Z",
        },
        "retrieved_tool_audit": {
            "tool_calls": [
                {
                    "tool": "MarketFeatureTool",
                    "status": "SUCCESS",
                    "source_refs": ["postgresql.real_time_prices:AAPL:2026-05-13T15:45:00Z"],
                    "result_hash": "sha256:market001",
                }
            ],
            "tool_result_bundle_hash": "sha256:tools001",
        },
        "data_citations": ["postgresql.real_time_prices:AAPL:2026-05-13T15:45:00Z"],
        "debate_applied": False,
        "debate_summary": None,
        "bullish_critic_points": [],
        "bearish_critic_points": [],
        "not_financial_advice": True,
        "symbol": "AAPL",
        "recommendation": "HOLD",
        "time_horizon": "SHORT_TERM",
        "summary": "AAPL is rated HOLD due to mixed technical and valuation signals.",
        "decision_rationale": [
            {
                "factor": "market_signal",
                "stance": "BULLISH",
                "weight": "MEDIUM",
                "explanation": "Technical indicators are mildly positive.",
            },
            {
                "factor": "valuation",
                "stance": "NEUTRAL",
                "weight": "MEDIUM",
                "explanation": "The stock is fairly valued relative to sector benchmarks.",
            },
        ],
        "supporting_signals": ["ML probability is modestly positive"],
        "conflicting_signals": ["Valuation upside is limited"],
        "conflict_level": "MEDIUM",
        "risk_warnings": ["Technology sector concentration can increase volatility."],
        "limitations": [],
        "source_quality": {
            "overall_quality_score": 0.87,
            "freshness_score": 0.95,
            "relevance_score": 0.78,
            "completeness_score": 0.86,
        },
    }
