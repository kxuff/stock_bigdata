from datetime import UTC, datetime

from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle
from tools.run_upstream_advisory import build_tool_result_bundle


def _request(symbols: list[str]) -> AdvisoryDecisionRequest:
    now = datetime(2026, 5, 29, tzinfo=UTC)
    return AdvisoryDecisionRequest.model_validate(
        {
            "request_id": "req-1",
            "timestamp": now.isoformat(),
            "as_of_timestamp": now.isoformat(),
            "user_query": "advise",
            "decision_mode": "portfolio_recommendation",
            "symbols": symbols,
        }
    )


def test_sentiment_source_refs_prefer_sentiment_only_fallbacks() -> None:
    bundle = build_tool_result_bundle(
        _request(["AAA", "BBB", "CCC"]),
        [
            {"Symbol": "AAA", "sentiment_score": 0.2, "sentiment_source_refs": ["sentiment:aaa"]},
            {"Symbol": "BBB", "sentiment_score": 0.1, "source_refs_x": ["sentiment:bbb"]},
            {"Symbol": "CCC", "sentiment_score": 0.0, "source_refs": ["shared:ccc"]},
        ],
        "base",
    )
    ToolResultBundle.model_validate(bundle)

    assert bundle["sentiment_snapshot"]["source_refs"] == ["sentiment:aaa", "sentiment:bbb", "shared:ccc"]


def test_valuation_source_refs_prefer_valuation_only_fallbacks() -> None:
    bundle = build_tool_result_bundle(
        _request(["AAA", "BBB", "CCC"]),
        [
            {"Symbol": "AAA", "valuation_label": "UNDERVALUED", "valuation_source_refs": ["valuation:aaa"]},
            {"Symbol": "BBB", "valuation_label": "FAIRLY_VALUED", "source_refs_y": ["valuation:bbb"]},
            {"Symbol": "CCC", "valuation_label": "UNKNOWN", "source_refs": ["shared:ccc"]},
        ],
        "base",
    )
    ToolResultBundle.model_validate(bundle)

    assert bundle["valuation_snapshot"]["source_refs"] == ["valuation:aaa", "valuation:bbb", "shared:ccc"]


def test_pred_a_drives_probability_final_score_drives_trend() -> None:
    bundle = build_tool_result_bundle(
        _request(["AAA"]),
        [{"Symbol": "AAA", "pred_a": 0.8, "final_score": -0.4, "risk_prob": 0.9}],
        "base",
    )
    ToolResultBundle.model_validate(bundle)

    assert bundle["ml_predictions"]["data"]["AAA"]["probability_up"] == 0.8
    assert bundle["market_features"]["data"]["AAA"]["trend_direction"] == "DOWN"


def test_missing_risk_prob_uses_neutral_schema_fallback_without_damping_ml_probability() -> None:
    bundle = build_tool_result_bundle(
        _request(["AAA"]),
        [{"Symbol": "AAA", "pred_a": 0.8, "final_score": -0.4}],
        "base",
    )
    ToolResultBundle.model_validate(bundle)

    ml = bundle["ml_predictions"]["data"]["AAA"]
    risk = bundle["risk_snapshot"]["data"]["AAA"]
    assert ml["probability_up"] == 0.8
    assert "risk_prob unavailable; using neutral fallback" in risk["risk_factors"]
    assert risk["risk_factors"][0] == "upstream risk_prob=0.500"
