from typing import Any

from app.schemas.agent_outputs import MarketDataAgentOutput, MarketSignal
from app.schemas.enums import AgentStatus, SignalStance, ToolStatus
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle


def analyze_market_data(
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
) -> MarketDataAgentOutput:
    market_result = tool_results.market_features
    ml_result = tool_results.ml_predictions
    source_refs = []
    missing_fields: list[str] = []
    limitations: list[str] = []

    if market_result is None or market_result.status != ToolStatus.SUCCESS:
        return MarketDataAgentOutput(
            status=AgentStatus.ERROR,
            summary="Required market feature context is unavailable.",
            confidence=0.0,
            missing_fields=["market_features"],
            limitations=["MARKET_CONTEXT_UNAVAILABLE"],
            source_refs=[],
            market_signals=[],
            ml_signal_available=False,
        )

    source_refs.extend(market_result.source_refs)
    if ml_result is not None:
        source_refs.extend(ml_result.source_refs)

    market_signals: list[MarketSignal] = []
    ml_signal_available = ml_result is not None and ml_result.status == ToolStatus.SUCCESS
    for symbol in request.symbols:
        market_feature = market_result.data.get(symbol)
        if market_feature is None:
            missing_fields.append(f"market_features.{symbol}")
            continue

        ml_prediction = ml_result.data.get(symbol) if ml_signal_available and ml_result else None
        if ml_prediction is None:
            missing_fields.append(f"ml_predictions.{symbol}")
            limitations.append(f"ML signal unavailable for {symbol}.")

        stance = _market_stance(market_feature.trend_direction, ml_prediction)
        confidence = _market_confidence(market_feature.trend_direction, ml_prediction)
        drivers = [
            f"trend_direction={market_feature.trend_direction}",
            f"price_change_pct_1d={market_feature.price_change_pct_1d}",
        ]
        if market_feature.technical_indicators.rsi_14 is not None:
            drivers.append(f"rsi_14={market_feature.technical_indicators.rsi_14}")
        if ml_prediction is not None:
            drivers.append(f"ml_probability_up={ml_prediction.probability_up}")

        market_signals.append(
            MarketSignal(
                symbol=symbol,
                stance=stance,
                confidence=confidence,
                drivers=drivers,
            )
        )

    if not market_signals:
        return MarketDataAgentOutput(
            status=AgentStatus.ERROR,
            summary="No requested symbols had market feature context.",
            confidence=0.0,
            missing_fields=missing_fields,
            limitations=["MARKET_SYMBOL_CONTEXT_UNAVAILABLE"],
            source_refs=source_refs,
            market_signals=[],
            ml_signal_available=ml_signal_available,
        )

    confidence = round(
        sum(signal.confidence for signal in market_signals) / len(market_signals),
        2,
    )
    return MarketDataAgentOutput(
        status=AgentStatus.DEGRADED if missing_fields else AgentStatus.SUCCESS,
        summary="Market and ML signals were summarized from upstream tool results.",
        confidence=confidence,
        missing_fields=missing_fields,
        limitations=limitations,
        source_refs=source_refs,
        market_signals=market_signals,
        ml_signal_available=ml_signal_available,
    )


def _market_stance(trend_direction: str, ml_prediction: Any | None) -> SignalStance:
    if ml_prediction is not None:
        if ml_prediction.probability_up >= 0.6:
            return SignalStance.BULLISH
        if ml_prediction.probability_down >= 0.6:
            return SignalStance.BEARISH

    normalized_trend = trend_direction.upper()
    if normalized_trend in {"UP", "BULLISH"}:
        return SignalStance.BULLISH
    if normalized_trend in {"DOWN", "BEARISH"}:
        return SignalStance.BEARISH
    return SignalStance.NEUTRAL


def _market_confidence(trend_direction: str, ml_prediction: Any | None) -> float:
    trend_confidence = 0.66 if trend_direction.upper() in {"UP", "DOWN"} else 0.55
    if ml_prediction is None:
        return trend_confidence
    directional_probability = max(ml_prediction.probability_up, ml_prediction.probability_down)
    return round((trend_confidence + directional_probability) / 2, 2)
