from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import AgentSettings
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle
from app.services.decision_service import AdvisoryDecisionService


def main() -> None:
    args = _parse_args()
    request_data = _read_json(args.request)
    request = AdvisoryDecisionRequest.model_validate(request_data)

    upstream = _load_upstream_rows(args.upstream)
    bundle_data = build_tool_result_bundle(
        request,
        upstream,
        args.source_ref,
        include_optional_placeholders=args.include_optional_placeholders,
    )
    bundle = ToolResultBundle.model_validate(bundle_data)

    service = AdvisoryDecisionService(
        settings=AgentSettings(
            advisory_use_crewai_manager=False,
            advisory_output_dir=args.output_dir,
        )
    )
    decision = service.decide(request, bundle)
    print(decision.model_dump_json(indent=2))


def build_tool_result_bundle(
    request: AdvisoryDecisionRequest,
    upstream_rows: list[dict[str, Any]],
    source_ref: str,
    *,
    include_optional_placeholders: bool = False,
) -> dict[str, Any]:
    rows_by_symbol = {str(row.get("Symbol") or row.get("symbol") or "").upper(): row for row in upstream_rows}
    now = request.as_of_timestamp or datetime.now(UTC)
    freshness = {
        "is_stale": False,
        "last_updated_at": now.isoformat(),
        "max_age_seconds": 86400,
    }

    market_data: dict[str, Any] = {}
    ml_data: dict[str, Any] = {}
    risk_data: dict[str, Any] = {}
    sentiment_data: dict[str, Any] = {}
    valuation_data: dict[str, Any] = {}
    source_refs = []
    sentiment_source_refs = []
    valuation_source_refs = []

    for symbol in request.symbols:
        row = rows_by_symbol.get(symbol, {})
        close = _float(row, "Close", "close", "latest_price", default=1.0)
        pred_a = _float(row, "pred_a", "prediction", "final_score", default=0.0)
        risk_prob = _float(row, "risk_prob", "probability_down", default=0.5)
        final_score = _float(row, "final_score", default=pred_a * (1 - risk_prob))
        probability_up = _probability_from_score(final_score, risk_prob)
        probability_down = max(0.0, min(1.0, 1.0 - probability_up))
        risk_label = _risk_label(risk_prob)
        row_source_ref = str(row.get("source_ref") or f"{source_ref}:{symbol}")
        source_refs.append(row_source_ref)
        max_drawdown_90d = _float(row, "maxdd90", "max_drawdown_90d", default=_float(row, "maxdd20", default=0.0))
        risk_window = str(row.get("risk_window") or ("90d" if "maxdd90" in row else "20d_proxy"))

        market_data[symbol] = {
            "latest_price": max(close, 0.01),
            "price_change_pct_1d": _float(row, "r1", "price_change_pct_1d", default=0.0) * 100,
            "volume_ratio_20d": max(_float(row, "RVOL20", "volume_ratio_20d", default=1.0), 0.0),
            "trend_direction": "UP" if final_score > 0 else "DOWN" if final_score < 0 else "FLAT",
            "technical_indicators": {
                "rsi_14": _nullable_float(row, "RSI14", "rsi_14"),
                "macd_signal": "BULLISH" if _float(row, "MACD_hist", default=0.0) >= 0 else "BEARISH",
                "bollinger_position": "MIDDLE",
                "sma20_vs_price": "ABOVE" if _float(row, "dist_ema20", default=0.0) >= 0 else "BELOW",
            },
        }
        ml_data[symbol] = {
            "predicted_direction": "UP" if probability_up >= probability_down else "DOWN",
            "probability_up": probability_up,
            "probability_down": probability_down,
            "model_version": str(row.get("model_version") or "upstream-ml"),
            "feature_window": str(row.get("feature_version") or "price_v1"),
        }
        risk_data[symbol] = {
            "risk_label": risk_label,
            "volatility_30d": max(_float(row, "vol20", "volatility_30d", default=risk_prob), 0.0),
            "max_drawdown_90d": max_drawdown_90d,
            "beta": _nullable_float(row, "beta_60D", "beta"),
            "risk_factors": [f"upstream risk_prob={risk_prob:.3f}", f"drawdown_window={risk_window}"],
            "confidence_cap": max(0.25, min(0.9, 1.0 - risk_prob / 2)),
        }
        if _has_any(row, "sentiment_score", "sentiment_label", "article_count", "top_drivers"):
            sentiment_source_refs.extend(_list_value(_first(row, "sentiment_source_ref", "source_refs_x")))
            score = max(-1.0, min(1.0, _float(row, "sentiment_score", default=0.0)))
            sentiment_data[symbol] = {
                "sentiment_label": str(_first(row, "sentiment_label") or _sentiment_label(score)),
                "sentiment_score": score,
                "article_count": int(_float(row, "article_count", default=0.0)),
                "top_drivers": _list_value(row.get("top_drivers")),
            }
        elif include_optional_placeholders:
            sentiment_data[symbol] = {
                "sentiment_label": "NEUTRAL",
                "sentiment_score": 0.0,
                "article_count": 0,
                "top_drivers": ["sentiment upstream unavailable; neutral fallback"],
            }

        if _has_any(row, "valuation_label", "pe_ratio", "sector_pe_ratio", "fair_value_estimate", "upside_downside_pct"):
            valuation_source_refs.extend(_list_value(_first(row, "valuation_source_ref", "source_refs_y")))
            valuation_data[symbol] = {
                "valuation_label": str(_first(row, "valuation_label") or "UNKNOWN"),
                "pe_ratio": _nullable_float(row, "pe_ratio"),
                "sector_pe_ratio": _nullable_float(row, "sector_pe_ratio"),
                "fair_value_estimate": _nullable_float(row, "fair_value_estimate"),
                "upside_downside_pct": _nullable_float(row, "upside_downside_pct"),
            }
        elif include_optional_placeholders:
            valuation_data[symbol] = {
                "valuation_label": "UNKNOWN",
                "pe_ratio": None,
                "sector_pe_ratio": None,
                "fair_value_estimate": None,
                "upside_downside_pct": final_score * 100,
            }

    base = {
        "status": "SUCCESS",
        "request_id": request.request_id,
        "as_of_timestamp": now.isoformat(),
        "freshness": freshness,
        "source_refs": source_refs or [source_ref],
    }
    sentiment_base = {**base, "source_refs": _unique(sentiment_source_refs) or base["source_refs"]}
    valuation_base = {**base, "source_refs": _unique(valuation_source_refs) or base["source_refs"]}
    bundle = {
        "market_features": {**base, "tool": "MarketFeatureTool", "data": market_data},
        "ml_predictions": {**base, "tool": "MlPredictionTool", "data": ml_data},
        "risk_snapshot": {**base, "tool": "RiskFeatureTool", "data": risk_data},
    }
    if sentiment_data:
        bundle["sentiment_snapshot"] = {**sentiment_base, "tool": "NewsSentimentTool", "data": sentiment_data}
    if valuation_data:
        bundle["valuation_snapshot"] = {**valuation_base, "tool": "FundamentalsTool", "data": valuation_data}
    return bundle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ORCA advisory from upstream ML output.")
    parser.add_argument("--request", type=Path, required=True, help="ORCA AdvisoryDecisionRequest JSON.")
    parser.add_argument("--upstream", type=Path, required=True, help="Upstream predictions/features JSON, CSV, or parquet.")
    parser.add_argument("--source-ref", default="stock_bigdata.ml_ready.stock_predictions", help="Citation source ref.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/advisory_decisions"))
    parser.add_argument(
        "--include-optional-placeholders",
        action="store_true",
        help="Emit neutral/unknown sentiment and valuation placeholders. Default omits them.",
    )
    return parser.parse_args()


def _load_upstream_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = _read_json(path)
        if isinstance(data, dict):
            data = data.get("rows", [data])
        return list(data)
    if suffix == ".csv":
        import pandas as pd

        return pd.read_csv(path).to_dict(orient="records")
    if suffix in {".parquet", ".pq"}:
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported upstream file: {path}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(row: dict[str, Any], *keys: str, default: float) -> float:
    value = _first(row, *keys)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _nullable_float(row: dict[str, Any], *keys: str) -> float | None:
    value = _first(row, *keys)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None and not _is_nan(row[key]):
            return row[key]
    return None


def _is_nan(value: Any) -> bool:
    try:
        return value != value
    except Exception:
        return False


def _has_any(row: dict[str, Any], *keys: str) -> bool:
    return any(_first(row, key) is not None for key in keys)


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if str(item).strip()]
            except ValueError:
                pass
        return [stripped]
    return [str(value)]


def _sentiment_label(score: float) -> str:
    if score >= 0.2:
        return "BULLISH"
    if score <= -0.2:
        return "BEARISH"
    return "NEUTRAL"


def _unique(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _probability_from_score(final_score: float, risk_prob: float) -> float:
    if 0 <= final_score <= 1:
        return max(0.0, min(1.0, final_score))
    base = 0.5 + max(-0.49, min(0.49, final_score))
    return max(0.0, min(1.0, base * (1 - risk_prob / 2)))


def _risk_label(risk_prob: float) -> str:
    if risk_prob >= 0.75:
        return "CRITICAL"
    if risk_prob >= 0.55:
        return "HIGH"
    if risk_prob >= 0.30:
        return "MEDIUM"
    return "LOW"


if __name__ == "__main__":
    main()
