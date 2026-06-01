from datetime import datetime

import pytest

from app.schemas.enums import DecisionMode, ToolStatus
from app.schemas.enums import RiskLabel, SentimentLabel, ValuationLabel
from app.schemas.request import AdvisoryDecisionRequest
from app.infrastructure.bigdata.bigdata_ml_provider import BigdataMlToolResultProvider


def _request(
    symbols: list[str], mode: DecisionMode = DecisionMode.SINGLE_SYMBOL_ADVISORY
) -> AdvisoryDecisionRequest:
    timestamp = datetime.fromisoformat("2026-05-27 23:30:00")
    return AdvisoryDecisionRequest(
        request_id="req_bigdata_ml_test",
        timestamp=timestamp,
        as_of_timestamp=timestamp,
        user_query="test bigdata ml provider",
        decision_mode=mode,
        symbols=symbols,
    )


def _adbe_row(close: float = 442.25) -> dict[str, object]:
    return {
        "Symbol": "ADBE",
        "Datetime": "2026-05-27 00:00:00",
        "model_version": "xgb_v1",
        "pred_a": 0.541499137878418,
        "risk_prob": 0.0377756766974926,
        "vol20": 0.021,
        "maxdd20": -0.045,
        "beta_60D": 1.12,
        "final_score": 0.51,
        "feature_version": "price_v1_notebook_ac",
        "prediction_process_date": "2026-05-27 23:12:00",
        "source_feature_process_date": "2026-05-27 22:59:00",
        "Close": close,
        "r1": 0.0125,
        "RVOL20": 1.34,
        "RSI14": 58.2,
        "MACD_hist": 0.17,
        "BB_pctB": 0.84,
        "BB_width": 0.11,
        "EMA20_50_spread": 2.1,
        "EMA20_slope": 0.4,
        "ROC10": 1.9,
        "ADX14": 24.0,
        "sentiment_label": "BULLISH",
        "sentiment_score": 0.42,
        "article_count": 7,
        "top_drivers": ["earnings", "AI demand"],
        "latest_article_published_at": "2026-05-27 18:00:00",
        "sentiment_scored_at": "2026-05-27 19:00:00",
        "sentiment_as_of_date": "2026-05-27",
        "valuation_label": "FAIRLY_VALUED",
        "pe_ratio": 32.5,
        "sector_pe_ratio": 30.0,
        "fair_value_estimate": 455.0,
        "upside_downside_pct": 0.029,
        "valuation_method": "relative_pe",
        "valuation_quality": "medium",
        "valuation_fetched_at": "2026-05-27 20:00:00",
        "valuation_as_of_date": "2026-05-27",
    }


def test_bigdata_ml_provider_returns_adbe_context() -> None:
    request = _request(["ADBE"])
    bundle = BigdataMlToolResultProvider(row_loader=lambda _: [_adbe_row()]).get_tool_results(request)

    assert bundle.ml_predictions is not None
    assert bundle.ml_predictions.status == ToolStatus.SUCCESS
    prediction = bundle.ml_predictions.data["ADBE"]
    assert prediction.predicted_direction == "NEUTRAL"
    assert prediction.probability_up == pytest.approx(0.541499137878418)
    assert prediction.probability_down == pytest.approx(1.0 - 0.541499137878418)
    assert prediction.model_version == "xgb_v1"
    assert prediction.feature_window == "price_v1_notebook_ac"

    assert bundle.market_features is not None
    assert bundle.market_features.status == ToolStatus.SUCCESS
    market_feature = bundle.market_features.data["ADBE"]
    assert market_feature.latest_price == pytest.approx(442.25)
    assert market_feature.price_change_pct_1d == pytest.approx(0.0125)
    assert market_feature.volume_ratio_20d == pytest.approx(1.34)
    assert market_feature.trend_direction == "UP"
    assert market_feature.technical_indicators.rsi_14 == pytest.approx(58.2)
    assert market_feature.technical_indicators.macd_signal == "BULLISH"
    assert market_feature.technical_indicators.bollinger_position == "UPPER"
    bundle.validate_required_for(request)
    assert bundle.market_features.source_refs == [
        "ml_ready.stock_price_features:ADBE:2026-05-27 00:00:00",
        "curated.us_stock_eod_prices:ADBE:2026-05-27 00:00:00",
    ]
    assert bundle.ml_predictions.source_refs == ["ml_ready.stock_predictions:ADBE:2026-05-27 00:00:00"]
    assert bundle.risk_snapshot is not None
    assert bundle.risk_snapshot.source_refs == [
        "ml_ready.stock_predictions:ADBE:2026-05-27 00:00:00",
        "ml_ready.stock_price_features:ADBE:2026-05-27 00:00:00",
    ]


def test_bigdata_ml_provider_missing_symbol_returns_unavailable() -> None:
    bundle = BigdataMlToolResultProvider(row_loader=lambda _: [_adbe_row()]).get_tool_results(
        _request(["ZZZZ"])
    )

    assert bundle.ml_predictions is not None
    assert bundle.ml_predictions.status == ToolStatus.UNAVAILABLE
    assert bundle.ml_predictions.error_message
    assert bundle.ml_predictions.data == {}
    assert bundle.market_features is not None
    assert bundle.market_features.status == ToolStatus.UNAVAILABLE
    assert bundle.market_features.error_message
    assert bundle.market_features.data == {}


def test_bigdata_ml_provider_maps_full_bundle_context() -> None:
    request = _request(["ADBE"])
    bundle = BigdataMlToolResultProvider(row_loader=lambda _: [_adbe_row()]).get_tool_results(request)

    assert bundle.risk_snapshot is not None
    risk = bundle.risk_snapshot.data["ADBE"]
    assert risk.risk_label == RiskLabel.LOW
    assert risk.volatility_30d == pytest.approx(0.021)
    assert risk.max_drawdown_90d == pytest.approx(-0.045)
    assert risk.beta == pytest.approx(1.12)
    assert risk.confidence_cap == pytest.approx(0.9)
    assert risk.risk_factors == ["upstream risk_prob=0.0378"]

    assert bundle.sentiment_snapshot is not None
    sentiment = bundle.sentiment_snapshot.data["ADBE"]
    assert sentiment.sentiment_label == SentimentLabel.BULLISH
    assert sentiment.sentiment_score == pytest.approx(0.42)
    assert sentiment.article_count == 7
    assert sentiment.top_drivers == ["earnings", "AI demand"]

    assert bundle.valuation_snapshot is not None
    valuation = bundle.valuation_snapshot.data["ADBE"]
    assert valuation.valuation_label == ValuationLabel.FAIRLY_VALUED
    assert valuation.pe_ratio == pytest.approx(32.5)
    assert valuation.fair_value_estimate == pytest.approx(455.0)


def test_bigdata_ml_provider_omits_optional_context_when_absent() -> None:
    row = _adbe_row()
    for key in list(row):
        if key.startswith("sentiment_") or key.startswith("valuation_"):
            row.pop(key)
    row.pop("article_count")
    row.pop("top_drivers")
    row.pop("pe_ratio")
    row.pop("sector_pe_ratio")
    row.pop("fair_value_estimate")
    row.pop("upside_downside_pct")

    bundle = BigdataMlToolResultProvider(row_loader=lambda _: [row]).get_tool_results(_request(["ADBE"]))

    assert bundle.risk_snapshot is not None
    assert bundle.sentiment_snapshot is None
    assert bundle.valuation_snapshot is None


@pytest.mark.parametrize(
    ("risk_prob", "label", "cap"),
    [(0.35, RiskLabel.MEDIUM, 0.825), (0.6, RiskLabel.HIGH, 0.7), (0.75, RiskLabel.CRITICAL, 0.625)],
)
def test_bigdata_ml_provider_maps_risk_thresholds(risk_prob: float, label: RiskLabel, cap: float) -> None:
    row = _adbe_row()
    row["risk_prob"] = risk_prob
    row["vol20"] = 0.05
    row["maxdd20"] = -0.12

    bundle = BigdataMlToolResultProvider(row_loader=lambda _: [row]).get_tool_results(_request(["ADBE"]))

    assert bundle.risk_snapshot is not None
    risk = bundle.risk_snapshot.data["ADBE"]
    assert risk.risk_label == label
    assert risk.confidence_cap == pytest.approx(cap)
    assert "high volatility vol20=0.0500" in risk.risk_factors
    assert "large drawdown maxdd20=-0.1200" in risk.risk_factors


def test_bigdata_ml_provider_partial_two_symbols() -> None:
    request = _request(["ADBE", "MSFT"], DecisionMode.PORTFOLIO_RECOMMENDATION)
    bundle = BigdataMlToolResultProvider(row_loader=lambda _: [_adbe_row()]).get_tool_results(request)

    assert bundle.ml_predictions is not None
    assert bundle.ml_predictions.status == ToolStatus.PARTIAL
    assert set(bundle.ml_predictions.data) == {"ADBE"}
    assert bundle.market_features is not None
    assert bundle.market_features.status == ToolStatus.PARTIAL
    assert set(bundle.market_features.data) == {"ADBE"}


def test_bigdata_ml_provider_invalid_close_skipped() -> None:
    bundle = BigdataMlToolResultProvider(row_loader=lambda _: [_adbe_row(close=0.0)]).get_tool_results(
        _request(["ADBE"])
    )

    assert bundle.ml_predictions is not None
    assert bundle.ml_predictions.status == ToolStatus.UNAVAILABLE
    assert bundle.ml_predictions.data == {}
    assert bundle.market_features is not None
    assert bundle.market_features.status == ToolStatus.UNAVAILABLE
    assert bundle.market_features.data == {}


def test_bigdata_ml_provider_marks_stale_context() -> None:
    row = _adbe_row()
    row["Datetime"] = "2026-05-25 00:00:00"
    row["prediction_process_date"] = "2026-05-25 23:12:00"
    row["source_feature_process_date"] = "2026-05-25 22:59:00"

    bundle = BigdataMlToolResultProvider(row_loader=lambda _: [row]).get_tool_results(_request(["ADBE"]))

    assert bundle.ml_predictions is not None
    assert bundle.ml_predictions.freshness.is_stale is True


def test_bigdata_ml_provider_marks_fresh_context() -> None:
    bundle = BigdataMlToolResultProvider(row_loader=lambda _: [_adbe_row()]).get_tool_results(_request(["ADBE"]))

    assert bundle.ml_predictions is not None
    assert bundle.ml_predictions.freshness.is_stale is False


def test_bigdata_ml_provider_handles_timezone_aware_request_freshness() -> None:
    request = _request(["ADBE"])
    request.as_of_timestamp = datetime.fromisoformat("2026-05-27T23:30:00+00:00")

    bundle = BigdataMlToolResultProvider(row_loader=lambda _: [_adbe_row()]).get_tool_results(request)

    assert bundle.ml_predictions is not None
    assert bundle.ml_predictions.freshness.is_stale is False


@pytest.mark.parametrize(("pred_a", "direction"), [(0.55, "UP"), (0.45, "DOWN"), (0.50, "NEUTRAL")])
def test_bigdata_ml_provider_direction_uses_pred_a_not_risk(pred_a: float, direction: str) -> None:
    row = _adbe_row()
    row["pred_a"] = pred_a
    row["risk_prob"] = 0.95

    bundle = BigdataMlToolResultProvider(row_loader=lambda _: [row]).get_tool_results(_request(["ADBE"]))

    assert bundle.ml_predictions is not None
    prediction = bundle.ml_predictions.data["ADBE"]
    assert prediction.predicted_direction == direction
    assert prediction.probability_down == pytest.approx(1.0 - pred_a)
