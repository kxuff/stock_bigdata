from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ORCA_MARKET_COLUMNS = [
    "Close",
    "r1",
    "RVOL20",
    "RSI14",
    "MACD_hist",
    "dist_ema20",
    "vol20",
    "maxdd20",
    "maxdd90",
    "beta_60D",
]


def write_orca_upstream_context(
    *,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    stage_path: Path,
    source_ref_prefix: str,
    sentiment_path: Path | None = None,
    valuation_path: Path | None = None,
) -> dict[str, Any]:
    """Write market/risk/ML context plus optional agent context for ORCA."""
    join_keys = ["Datetime", "Symbol", "feature_version"]
    feature_columns = [column for column in ORCA_MARKET_COLUMNS if column in features.columns]
    context = predictions.merge(
        features[[*join_keys, *feature_columns]],
        on=join_keys,
        how="left",
    )
    context["source_ref"] = context["Symbol"].map(lambda symbol: f"{source_ref_prefix}:{symbol}")
    if "maxdd90" in context.columns:
        context["risk_window"] = context["maxdd90"].notna().map(lambda has_90d: "90d" if has_90d else "20d_proxy")
    else:
        context["risk_window"] = "20d_proxy"
    context["market_context_symbol"] = "SPY"
    context["orca_context_version"] = "orca_market_risk_ml_v1"

    included = ["market_features", "ml_predictions", "risk_snapshot"]
    excluded = ["portfolio_snapshot"]
    sentiment = _read_optional_context(sentiment_path)
    if not sentiment.empty:
        sentiment = sentiment.rename(columns={"source_refs": "sentiment_source_refs"})
        context = context.merge(sentiment, on="Symbol", how="left")
        if context["sentiment_score"].notna().any():
            included.append("sentiment_snapshot")
        else:
            excluded.append("sentiment_snapshot")
    else:
        excluded.append("sentiment_snapshot")

    valuation = _read_optional_context(valuation_path)
    if not valuation.empty:
        valuation = valuation.rename(columns={"source_refs": "valuation_source_refs"})
        context = context.merge(valuation, on="Symbol", how="left")
        if context["valuation_label"].notna().any():
            included.append("valuation_snapshot")
        else:
            excluded.append("valuation_snapshot")
    else:
        excluded.append("valuation_snapshot")

    output_path = stage_path / "orca_upstream.json"
    context.to_json(output_path, orient="records", date_format="iso")
    return {
        "orca_upstream_context": str(output_path),
        "orca_context_rows": int(len(context)),
        "orca_context_version": "orca_market_risk_ml_v1",
        "orca_context_includes": included,
        "orca_context_excludes": excluded,
    }


def _read_optional_context(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if frame.empty or "Symbol" not in frame.columns:
        return pd.DataFrame()
    return frame
