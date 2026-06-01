from datetime import datetime
from typing import Any, Mapping

from app.schemas.tool_results import MarketFeature, MlPrediction, TechnicalIndicators


def is_valid_context_row(row: Mapping[str, Any]) -> bool:
    return _float(row.get("Close")) > 0


def prediction_from_row(row: Mapping[str, Any]) -> MlPrediction:
    pred_a = _clamp(_float(row.get("pred_a")))
    if pred_a >= 0.55:
        direction = "UP"
    elif pred_a <= 0.45:
        direction = "DOWN"
    else:
        direction = "NEUTRAL"
    return MlPrediction(
        predicted_direction=direction,
        probability_up=pred_a,
        probability_down=1.0 - pred_a,
        model_version=str(row.get("model_version") or "unknown"),
        feature_window=str(row.get("feature_version") or "unknown"),
    )


def market_feature_from_row(row: Mapping[str, Any], fallback_direction: str) -> MarketFeature:
    rsi = _float_or_none(row.get("RSI14"))
    return MarketFeature(
        latest_price=_float(row.get("Close")),
        price_change_pct_1d=_float(row.get("r1")),
        volume_ratio_20d=max(_float(row.get("RVOL20"), 1.0), 0.0),
        trend_direction=_trend_direction(row, fallback_direction),
        technical_indicators=TechnicalIndicators(
            rsi_14=rsi if rsi is not None and 0 <= rsi <= 100 else None,
            macd_signal=_macd_signal(row.get("MACD_hist")),
            bollinger_position=_bollinger_position(row.get("BB_pctB")),
        ),
    )


def row_sort_key(row: Mapping[str, Any]) -> tuple[datetime, datetime, datetime]:
    return (
        parse_datetime(row.get("prediction_process_date") or row.get("process_date")),
        parse_datetime(row.get("source_feature_process_date")),
        parse_datetime(row.get("Datetime")),
    )


def parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if not value:
        return datetime.min
    return datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))


def _trend_direction(row: Mapping[str, Any], fallback_direction: str) -> str:
    spread = _float(row.get("EMA20_50_spread"))
    slope = _float(row.get("EMA20_slope"))
    roc = _float(row.get("ROC10"))
    adx = _float(row.get("ADX14"))
    score = sum(value > 0 for value in (spread, slope, roc)) - sum(value < 0 for value in (spread, slope, roc))
    if adx >= 15 and score > 0:
        return "UP"
    if adx >= 15 and score < 0:
        return "DOWN"
    return fallback_direction


def _macd_signal(value: Any) -> str:
    macd = _float(value)
    if macd > 0:
        return "BULLISH"
    if macd < 0:
        return "BEARISH"
    return "NEUTRAL"


def _bollinger_position(value: Any) -> str:
    bb_pctb = _float(value, 0.5)
    if bb_pctb >= 0.8:
        return "UPPER"
    if bb_pctb <= 0.2:
        return "LOWER"
    return "MIDDLE"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
