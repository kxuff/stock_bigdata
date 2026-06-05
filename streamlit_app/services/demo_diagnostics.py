from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


TOOL_STATUS_COLUMNS = [
    "market_features",
    "ml_predictions",
    "risk_snapshot",
    "sentiment_snapshot",
    "valuation_snapshot",
]


def normalize_symbol(value: str | None) -> str:
    return str(value or "").strip().upper().replace(".", "-")


def parse_symbols(value: str | None) -> list[str]:
    symbols: list[str] = []
    for raw in str(value or "").split(","):
        symbol = normalize_symbol(raw)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def flatten_coverage_rows(payload: dict[str, Any] | None) -> pd.DataFrame:
    rows = []
    for row in (payload or {}).get("rows") or []:
        tools = row.get("tools") or {}
        out: dict[str, Any] = {
            "Symbol": row.get("symbol"),
            "Ready": bool(row.get("ready")),
            "Latest Timestamp": row.get("latest_timestamp"),
            "Warnings": join_warnings(row.get("warnings")),
        }
        for tool in TOOL_STATUS_COLUMNS:
            out[tool] = (tools.get(tool) or {}).get("status", "MISSING")
        rows.append(out)
    return pd.DataFrame(
        rows,
        columns=["Symbol", "Ready", "Latest Timestamp", *TOOL_STATUS_COLUMNS, "Warnings"],
    )


def coverage_by_symbol(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        normalize_symbol(row.get("symbol")): row
        for row in (payload or {}).get("rows") or []
        if normalize_symbol(row.get("symbol"))
    }


def ready_count(payload: dict[str, Any] | None) -> int:
    return sum(1 for row in (payload or {}).get("rows") or [] if row.get("ready"))


def latest_coverage_timestamp(payload: dict[str, Any] | None) -> str | None:
    values = [
        str(row.get("latest_timestamp"))
        for row in (payload or {}).get("rows") or []
        if row.get("latest_timestamp")
    ]
    if not values:
        return None
    return max(values)


def join_warnings(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if str(item))
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def warning_list(value: Any) -> list[str]:
    text = join_warnings(value)
    if not text:
        return []
    return [part.strip() for part in text.replace("\n", ";").split(";") if part.strip()]


def classify_issue(value: Any) -> str:
    text = join_warnings(value).lower()
    if not text:
        return "OK"
    if (
        "no module named 'pyspark'" in text
        or "no module named \"pyspark\"" in text
        or "pyspark" in text
        or "hadoop_home" in text
        or "hadoop.home.dir" in text
        or "winutils" in text
    ):
        return "Spark runtime missing"
    if "connection" in text or "offline" in text or "refused" in text or "timed out" in text:
        return "ORCA API offline"
    if "finbert" in text:
        return "FinBERT missing"
    if "llm" in text or "ninerouter" in text or "api key" in text or "authentication" in text:
        return "LLM key missing"
    if "no prediction rows" in text or "prediction rows matched" in text or "empty prediction" in text:
        return "No prediction rows"
    if "stale" in text or "max_age" in text:
        return "Stale data"
    if "missing" in text or "unavailable" in text:
        return "Data missing"
    return "Needs attention"


def issue_summary(values: list[Any]) -> str:
    labels = [classify_issue(value) for value in values if classify_issue(value) != "OK"]
    if not labels:
        return "OK"
    seen: list[str] = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    return ", ".join(seen[:3])


def coverage_warning(row: dict[str, Any] | None) -> str:
    if not row:
        return "No coverage row returned."
    warnings = warning_list(row.get("warnings"))
    if warnings:
        return "; ".join(warnings[:3])
    tools = row.get("tools") or {}
    failing = [
        f"{name}: {(tools.get(name) or {}).get('status', 'MISSING')}"
        for name in TOOL_STATUS_COLUMNS
        if (tools.get(name) or {}).get("status") not in {"SUCCESS", "MISSING"}
    ]
    return "; ".join(failing[:3])


def should_disable_quick_action(label: str, *, api_offline: bool, has_symbol: bool, coverage_ready: bool) -> bool:
    if api_offline:
        return True
    normalized_label = label.lower()
    if normalized_label.startswith("advise"):
        return not has_symbol or not coverage_ready
    if normalized_label.startswith("compare"):
        return not has_symbol
    return False


def format_time(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)
