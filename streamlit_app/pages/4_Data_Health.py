from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from services.advisory_api import api_base_url, fetch_advisory_picks, fetch_data_coverage, fetch_health, fetch_status
from services.demo_diagnostics import (
    TOOL_STATUS_COLUMNS,
    classify_issue,
    flatten_coverage_rows,
    issue_summary,
    latest_coverage_timestamp,
    parse_symbols,
    ready_count,
    warning_list,
)


DEFAULT_SYMBOLS = "AAPL,MSFT,NVDA"


st.set_page_config(page_title="Data Health", page_icon="+", layout="wide")


def _safe_call(label: str, func, *args, **kwargs) -> tuple[Any | None, str | None]:
    try:
        return func(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001 - diagnostics page must stay up.
        return None, f"{label}: {exc}"


def _status_style(value: Any) -> str:
    text = str(value or "").upper()
    if text == "SUCCESS" or value is True:
        return "background-color: rgba(34, 197, 94, .14); color: #bbf7d0"
    if text in {"STALE", "PARTIAL", "MISSING"}:
        return "background-color: rgba(245, 158, 11, .16); color: #fde68a"
    if text in {"ERROR", "UNAVAILABLE"} or value is False:
        return "background-color: rgba(239, 68, 68, .16); color: #fecaca"
    return ""


def _style_coverage(frame: pd.DataFrame):
    return frame.style.applymap(_status_style, subset=["Ready", *TOOL_STATUS_COLUMNS])


with st.sidebar:
    st.header("Data Health")
    symbol_text = st.text_area("Symbols", value=DEFAULT_SYMBOLS, height=80)
    limit = st.number_input("Pick limit", min_value=1, max_value=100, value=25, step=1)
    min_pred_a = st.number_input("Min Pred_A", min_value=-1.0, max_value=1.0, value=0.06, step=0.01)
    max_risk_prob = st.number_input("Max Risk Prob", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
    if st.button("Refresh diagnostics", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"ORCA API: {api_base_url()}")


symbols = parse_symbols(symbol_text)

st.title("Data Health")
st.caption("Backend status, per-symbol data coverage, and AI pick availability for the local demo.")

health, health_error = _safe_call("healthz", fetch_health, timeout=3.0)
status, status_error = _safe_call("status", fetch_status, timeout=3.0)
coverage, coverage_error = (None, "No symbols provided.")
if symbols:
    coverage, coverage_error = _safe_call("coverage", fetch_data_coverage, symbols, timeout=30.0)
picks, picks_error = _safe_call(
    "picks",
    fetch_advisory_picks,
    limit=int(limit),
    min_pred_a=float(min_pred_a),
    max_risk_prob=float(max_risk_prob),
    timeout=30.0,
)

backend_connected = not health_error and not status_error
coverage_frame = flatten_coverage_rows(coverage if isinstance(coverage, dict) else {})
picks_warnings = warning_list((picks or {}).get("warnings")) if isinstance(picks, dict) else []
picks_count = int((picks or {}).get("count", 0)) if isinstance(picks, dict) else 0

summary_cols = st.columns(4)
summary_cols[0].metric("Backend", "Connected" if backend_connected else "Offline")
summary_cols[1].metric("Ready symbols", f"{ready_count(coverage)}/{len(symbols)}")
summary_cols[2].metric("Latest data", latest_coverage_timestamp(coverage) or "n/a")
summary_cols[3].metric("Qualified picks", picks_count)

errors = [error for error in [health_error, status_error, coverage_error, picks_error] if error]
warnings = []
if coverage_error:
    warnings.append(coverage_error)
if picks_warnings:
    warnings.extend(picks_warnings)
if errors:
    st.error(issue_summary(errors))
    with st.expander("Diagnostic errors", expanded=True):
        for error in errors:
            st.code(error, language="text")
elif picks_warnings:
    st.warning(issue_summary(list(picks_warnings)))

st.subheader("Backend")
backend_rows = [
    {"Check": "healthz", "State": "OK" if health and not health_error else "ERROR", "Detail": health or health_error},
    {"Check": "status", "State": "OK" if status and not status_error else "ERROR", "Detail": status or status_error},
]
st.dataframe(pd.DataFrame(backend_rows), use_container_width=True, hide_index=True)

st.subheader("Coverage")
if coverage_frame.empty:
    st.info("No coverage rows returned.")
else:
    st.dataframe(_style_coverage(coverage_frame), use_container_width=True, hide_index=True)

st.subheader("Picks")
if not isinstance(picks, dict):
    st.info("No picks payload returned.")
else:
    data = pd.DataFrame(picks.get("data") or [])
    if data.empty:
        issue = classify_issue(picks_warnings or picks_error or "No prediction rows matched the requested filters.")
        st.warning(f"{issue}: no qualified picks are available for the current filters.")
    else:
        st.dataframe(data, use_container_width=True, hide_index=True)
    if picks_warnings:
        with st.expander("Pick warnings", expanded=True):
            for warning in picks_warnings:
                st.code(str(warning), language="text")
