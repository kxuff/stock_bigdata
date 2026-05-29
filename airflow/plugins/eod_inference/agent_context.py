from __future__ import annotations

import math
import json
import os
from datetime import timedelta
from typing import Any
from urllib import request as urlrequest

import pandas as pd
import yfinance as yf

from eod_inference.config import PipelineConfig
from eod_inference.utils import parse_date, stage_dir, write_json


POSITIVE_TERMS = {
    "beat", "beats", "bullish", "buy", "growth", "gain", "gains", "higher", "outperform",
    "positive", "profit", "profits", "record", "rebound", "upgrade", "upside", "strong",
}
NEGATIVE_TERMS = {
    "bearish", "cut", "decline", "downgrade", "drop", "falls", "fall", "loss", "losses",
    "lower", "miss", "misses", "negative", "pressure", "risk", "sell", "slowdown", "weak",
}
SECTOR_PE_FALLBACK = {
    "Technology": 30.0,
    "Communication Services": 24.0,
    "Consumer Cyclical": 22.0,
    "Consumer Defensive": 21.0,
    "Healthcare": 20.0,
    "Financial Services": 13.0,
    "Industrials": 22.0,
    "Energy": 12.0,
    "Utilities": 18.0,
    "Real Estate": 18.0,
    "Basic Materials": 16.0,
}
FINBERT_FAILURE_LIMIT = int(os.getenv("FINBERT_FAILURE_LIMIT", "3"))


def build_agent_context(as_of_date: str) -> dict[str, Any]:
    config = PipelineConfig.from_env()
    target_date = parse_date(as_of_date)
    stage = stage_dir(config, target_date)
    symbols = config.symbols

    sentiment = _build_sentiment(symbols, target_date)
    valuation = _build_valuation(symbols, target_date)

    sentiment_path = stage / "sentiment_context.parquet"
    valuation_path = stage / "valuation_context.parquet"
    sentiment.to_parquet(sentiment_path, index=False)
    valuation.to_parquet(valuation_path, index=False)

    manifest = {
        "as_of_date": target_date.isoformat(),
        "sentiment_context": str(sentiment_path),
        "sentiment_rows": int(len(sentiment)),
        "valuation_context": str(valuation_path),
        "valuation_rows": int(len(valuation)),
    }
    write_json(stage / "agent_context_manifest.json", manifest)
    return manifest


def _build_sentiment(symbols: list[str], target_date) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    min_dt = pd.Timestamp(target_date - timedelta(days=7))
    max_dt = pd.Timestamp(target_date + timedelta(days=1))
    finbert_disabled = False
    finbert_failures = 0
    for symbol in symbols:
        articles = _news_items(symbol)
        scored: list[dict[str, Any]] = []
        fallback_scored: list[dict[str, Any]] = []
        finbert_used = False
        for item in articles:
            item = _flatten_news_item(item)
            published = _published_at(item)
            in_window = published is None or (min_dt <= published.tz_localize(None) < max_dt)
            title = str(item.get("title") or item.get("headline") or "").strip()
            summary = str(item.get("summary") or item.get("description") or "").strip()
            text = f"{title} {summary}".strip()
            if not text:
                continue
            finbert = None if finbert_disabled else _score_sentiment_with_finbert(text)
            if os.getenv("FINBERT_API_URL") and finbert is None:
                finbert_failures += 1
                if finbert_failures >= FINBERT_FAILURE_LIMIT:
                    finbert_disabled = True
            score = finbert["sentiment_score"] if finbert else _score_sentiment(text)
            candidate = {
                "headline": title or text[:180],
                "score": score,
                "url": item.get("link") or item.get("url"),
                "published_at": published,
            }
            if finbert:
                candidate["finbert_label"] = finbert.get("label")
                finbert_used = True
                finbert_failures = 0
            fallback_scored.append(candidate)
            if published is not None and not (min_dt <= published.tz_localize(None) < max_dt):
                continue
            scored.append(candidate)
        if not scored:
            scored = fallback_scored[:10]
        if not scored:
            continue
        article_count = len(scored)
        sentiment_score = float(sum(item["score"] for item in scored) / article_count)
        published_values = [item["published_at"].tz_localize(None) for item in scored if item.get("published_at") is not None]
        stale_article_count = sum(
            1
            for item in scored
            if item.get("published_at") is not None and item["published_at"].tz_localize(None) < min_dt
        )
        scored_at = pd.Timestamp.utcnow().tz_localize(None)
        rows.append(
            {
                "as_of_date": target_date.isoformat(),
                "Symbol": symbol,
                "sentiment_score": max(-1.0, min(1.0, sentiment_score)),
                "sentiment_label": _sentiment_label(sentiment_score, scored),
                "article_count": article_count,
                "latest_article_published_at": max(published_values) if published_values else pd.NaT,
                "oldest_article_published_at": min(published_values) if published_values else pd.NaT,
                "sentiment_scored_at": scored_at,
                "stale_article_count": stale_article_count,
                "top_drivers": [item["headline"] for item in sorted(scored, key=lambda row: abs(row["score"]), reverse=True)[:3]],
                "source_refs": _sentiment_source_refs(symbol, finbert_used=finbert_used),
                "process_date": scored_at,
            }
        )
    return pd.DataFrame(rows)


def _build_valuation(symbols: list[str], target_date) -> pd.DataFrame:
    raw_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        info = _ticker_info(symbol)
        if not info:
            continue
        current_price = _num(info.get("currentPrice") or info.get("regularMarketPrice"))
        trailing_pe = _num(info.get("trailingPE"))
        forward_pe = _num(info.get("forwardPE"))
        pe_ratio = trailing_pe or forward_pe
        target_mean_price = _num(info.get("targetMeanPrice"))
        sector = str(info.get("sector") or "").strip() or "Unknown"
        raw_rows.append(
            {
                "as_of_date": target_date.isoformat(),
                "Symbol": symbol,
                "sector": sector,
                "current_price": current_price,
                "pe_ratio": pe_ratio,
                "target_mean_price": target_mean_price,
                "analyst_count": _num(info.get("numberOfAnalystOpinions")),
                "source_refs": [f"yfinance.fundamentals:{symbol}"],
            }
        )

    if not raw_rows:
        return pd.DataFrame()
    raw = pd.DataFrame(raw_rows)
    sector_medians = raw.dropna(subset=["pe_ratio"]).groupby("sector")["pe_ratio"].median().to_dict()
    output_rows: list[dict[str, Any]] = []
    for row in raw.to_dict(orient="records"):
        sector_sample_count = int(raw[(raw["sector"] == row["sector"]) & raw["pe_ratio"].notna()].shape[0])
        sector_pe = sector_medians.get(row["sector"])
        valuation_method = "analyst_target" if row["target_mean_price"] else "sector_pe_relative"
        if not sector_pe:
            sector_pe = SECTOR_PE_FALLBACK.get(row["sector"])
            valuation_method = "fallback_sector_pe"
        pe_ratio = row["pe_ratio"]
        current_price = row["current_price"]
        fair_value = row["target_mean_price"]
        if fair_value is None and pe_ratio and sector_pe and current_price:
            fair_value = current_price * sector_pe / pe_ratio
        upside = None
        if fair_value and current_price:
            upside = (fair_value - current_price) / current_price * 100
        output_rows.append(
            {
                "as_of_date": row["as_of_date"],
                "Symbol": row["Symbol"],
                "valuation_label": _valuation_label(upside),
                "pe_ratio": pe_ratio,
                "sector_pe_ratio": sector_pe,
                "fair_value_estimate": fair_value,
                "upside_downside_pct": upside,
                "valuation_method": valuation_method if fair_value is not None else "unavailable",
                "valuation_quality": _valuation_quality(
                    valuation_method=valuation_method,
                    fair_value=fair_value,
                    pe_ratio=pe_ratio,
                    sector_sample_count=sector_sample_count,
                    analyst_count=row.get("analyst_count"),
                ),
                "valuation_fetched_at": pd.Timestamp.utcnow().tz_localize(None),
                "fundamentals_as_of": target_date.isoformat(),
                "sector_sample_count": sector_sample_count,
                "analyst_count": row.get("analyst_count"),
                "source_refs": row["source_refs"],
                "process_date": pd.Timestamp.utcnow().tz_localize(None),
            }
        )
    return pd.DataFrame(output_rows)


def _news_items(symbol: str) -> list[dict[str, Any]]:
    try:
        return list(yf.Ticker(symbol).news or [])
    except Exception:
        return []


def _flatten_news_item(item: dict[str, Any]) -> dict[str, Any]:
    content = item.get("content") if isinstance(item, dict) else None
    if not isinstance(content, dict):
        return item
    canonical = content.get("canonicalUrl") if isinstance(content.get("canonicalUrl"), dict) else {}
    return {
        **item,
        **content,
        "url": canonical.get("url") or content.get("url") or item.get("url"),
    }


def _ticker_info(symbol: str) -> dict[str, Any]:
    try:
        return dict(yf.Ticker(symbol).info or {})
    except Exception:
        return {}


def _published_at(item: dict[str, Any]) -> pd.Timestamp | None:
    value = item.get("providerPublishTime") or item.get("pubDate")
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return pd.to_datetime(value, unit="s", utc=True)
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def _score_sentiment(text: str) -> float:
    words = [word.strip(".,:;!?()[]{}\"'").lower() for word in text.split()]
    pos = sum(word in POSITIVE_TERMS for word in words)
    neg = sum(word in NEGATIVE_TERMS for word in words)
    if pos == 0 and neg == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / math.sqrt(pos + neg)))


def _score_sentiment_with_finbert(text: str) -> dict[str, Any] | None:
    api_url = os.getenv("FINBERT_API_URL", "").rstrip("/")
    if not api_url:
        return None
    endpoint = f"{api_url}/sentiment"
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urlrequest.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "1"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=float(os.getenv("FINBERT_API_TIMEOUT", "3"))) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    score = _num_with_zero(data.get("sentiment_score"))
    if score is None:
        positive = _num_with_zero(data.get("positive_prob")) or 0.0
        negative = _num_with_zero(data.get("negative_prob")) or 0.0
        score = positive - negative
    return {
        "label": str(data.get("label") or "").lower() or None,
        "sentiment_score": max(-1.0, min(1.0, float(score))),
    }


def _sentiment_source_refs(symbol: str, *, finbert_used: bool) -> list[str]:
    refs = [f"yfinance.news:{symbol}"]
    if finbert_used:
        refs.append("finbert:ProsusAI/finbert")
    return refs


def _sentiment_label(score: float, scored: list[dict[str, Any]]) -> str:
    has_pos = any(item["score"] > 0.2 for item in scored)
    has_neg = any(item["score"] < -0.2 for item in scored)
    if has_pos and has_neg:
        return "MIXED"
    if score >= 0.2:
        return "BULLISH"
    if score <= -0.2:
        return "BEARISH"
    return "NEUTRAL"


def _valuation_label(upside: float | None) -> str:
    if upside is None:
        return "UNKNOWN"
    if upside >= 10:
        return "UNDERVALUED"
    if upside <= -10:
        return "OVERVALUED"
    return "FAIRLY_VALUED"


def _valuation_quality(
    *,
    valuation_method: str,
    fair_value: float | None,
    pe_ratio: float | None,
    sector_sample_count: int,
    analyst_count: float | None,
) -> str:
    if fair_value is None:
        return "UNKNOWN"
    if valuation_method == "analyst_target" and analyst_count and analyst_count >= 5:
        return "MEDIUM"
    if valuation_method == "sector_pe_relative" and pe_ratio and sector_sample_count >= 10:
        return "MEDIUM"
    return "LOW"


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number) or number <= 0:
        return None
    return number


def _num_with_zero(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number
