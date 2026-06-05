from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from chat import state as chat_state
from services.advisory_api import create_agent_query_job, fetch_advisory_pick_detail, fetch_data_coverage
from services.backtest_api import DEFAULT_SYMBOLS, build_equity_figure, run_strategy_backtest
from services.demo_diagnostics import (
    classify_issue,
    coverage_by_symbol,
    coverage_warning as describe_coverage_warning,
    issue_summary,
    normalize_symbol,
)
from services.ml_inference_api import MAX_RISK_PROB_PCT, MIN_PRED_A, fetch_ml_inference_picks_result


st.set_page_config(page_title="AI Stock Picks", page_icon="*", layout="wide")
chat_state.init()
BACKTEST_CACHE_VERSION = "cr_metric_v2"

st.title("AI Stock Picks")
st.caption("Ranked ORCA model signals from the latest EOD inference batch.")

with st.expander("Output contract"):
    st.markdown(
        f"""
        - `Date`: signal date.
        - `Ticker`: stock symbol.
        - `Entry_Price`: proposed entry/reference price.
        - `Pred_A`: model upside signal as a decimal.
        - `Risk_Prob_%`: model risk probability as a percent.
        - `FinalScore`: `Pred_A * (1 - RiskProb)`.
        - `Ready`: required market, ML, and risk context is complete for demo use.
        - The page filters to `Pred_A >= {MIN_PRED_A:.2f}` and `Risk_Prob <= {MAX_RISK_PROB_PCT / 100:.2f}`.
        """
    )

try:
    picks_result = fetch_ml_inference_picks_result(limit=100)
    picks = picks_result.frame
    picks_warnings = list(picks_result.warnings)
    picks_source = picks_result.source
    picks_error = picks_result.error
except Exception as exc:  # noqa: BLE001 - UI should stay usable when backend/local data is absent.
    picks_warnings = [str(exc)]
    picks_source = "error"
    picks_error = str(exc)
    st.warning(f"Cannot load ML inference output: {classify_issue(str(exc))}: {exc}")
    picks = None


def _create_pick_job(symbol: str) -> None:
    payload = {
        "message": f"Should I buy {symbol} today?",
        "context": {
            "symbol": symbol,
            "investment_horizon": "SHORT_TERM",
            "risk_tolerance": "MODERATE",
        },
    }
    try:
        job = create_agent_query_job(payload, timeout=30.0)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not create ORCA job: {exc}")
        return
    chat_state.add_job(
        {
            "job_id": job.get("job_id"),
            "kind": "agent_query",
            "symbol": symbol,
            "prompt": payload["message"],
            "created_at": chat_state.utc_now().isoformat(),
            "updated_at": None,
            "status": job.get("status", "queued"),
            "result_fetched": False,
            "events_complete": False,
        }
    )
    st.success(f"ORCA job created: {job.get('job_id', 'unknown')}")
    st.page_link("pages/2_AI_Chat.py", label="Open AI Chat")


def _ready_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "ready"}


def _load_coverage(symbols: list[str]) -> tuple[dict[str, dict[str, Any]], str | None]:
    if not symbols:
        return {}, None
    try:
        payload = fetch_data_coverage(symbols, timeout=20.0)
    except Exception as exc:  # noqa: BLE001
        return {}, str(exc)
    return coverage_by_symbol(payload), None


def _render_pick_diagnostic(warnings: list[str], error: str | None = None) -> None:
    messages = [*warnings]
    if error and error not in messages:
        messages.insert(0, error)
    issue = issue_summary(messages or ["No prediction rows matched the requested filters."])
    st.warning(f"{issue}: no qualified ML picks are available for the current filters.")
    if messages:
        with st.expander("Diagnostics", expanded=True):
            for message in messages:
                st.code(str(message), language="text")


if picks is not None and not picks.empty:
    st.caption(f"Source: {picks_source}")
    if picks_warnings:
        st.warning(issue_summary(picks_warnings))
    dates = sorted(picks["Date"].dropna().unique(), reverse=True)
    tickers = sorted(picks["Ticker"].dropna().unique())
    col1, col2, col3, col4 = st.columns(4)
    selected_date = col1.selectbox("Signal date", dates)
    ticker_filter = col2.multiselect("Ticker", tickers, default=tickers)
    min_score = col3.slider("Minimum FinalScore", 0.0, float(max(MIN_PRED_A, picks["FinalScore"].max())), 0.0, 0.01)
    max_risk = col4.slider("Maximum risk %", 0.0, float(MAX_RISK_PROB_PCT), float(MAX_RISK_PROB_PCT), 0.5)

    filtered_picks = picks[
        (picks["Date"] == selected_date)
        & (picks["Ticker"].isin(ticker_filter))
        & (picks["FinalScore"] >= min_score)
        & (picks["Risk_Prob_%"].fillna(0) <= max_risk)
    ].sort_values("FinalScore", ascending=False)

    visible_symbols = sorted({normalize_symbol(symbol) for symbol in filtered_picks["Ticker"].dropna().astype(str).tolist()})
    coverage_map, coverage_error = _load_coverage(visible_symbols[:25])
    if coverage_error:
        st.warning(f"Coverage check failed: {classify_issue(coverage_error)}: {coverage_error}")

    ready_count = int(filtered_picks["Ready"].map(_ready_value).sum()) if "Ready" in filtered_picks.columns else len(filtered_picks)
    metrics = st.columns(4)
    metrics[0].metric("Signals", len(filtered_picks))
    metrics[1].metric("Ready", ready_count)
    metrics[2].metric("Best FinalScore", f"{filtered_picks['FinalScore'].max():.4f}" if not filtered_picks.empty else "n/a")
    metrics[3].metric("Average Pred_A", f"{filtered_picks['Pred_A'].mean() * 100:.2f}%" if not filtered_picks.empty else "n/a")

    st.subheader("Top Picks")
    for _, pick in filtered_picks.head(3).iterrows():
        symbol = normalize_symbol(str(pick["Ticker"]))
        coverage_row = coverage_map.get(symbol)
        ready = bool(coverage_row.get("ready")) if coverage_row else _ready_value(pick.get("Ready", True))
        warnings = describe_coverage_warning(coverage_row) if coverage_row else str(pick.get("Warnings") or "")
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.0, 0.8], vertical_alignment="center")
            c1.metric(symbol, f"${pick['Entry_Price']:,.2f}", f"FinalScore {pick['FinalScore']:.4f}")
            c2.metric("Pred_A", f"{pick['Pred_A'] * 100:.2f}%")
            c3.metric("Risk", f"{pick['Risk_Prob_%']:.2f}%")
            if c4.button("Ask ORCA", key=f"pick-job-{symbol}", disabled=not ready, use_container_width=True):
                _create_pick_job(symbol)
            if warnings:
                st.warning(warnings)

    st.subheader("Pick Detail")
    if visible_symbols:
        detail_symbol = st.selectbox("Detail symbol", visible_symbols, index=0)
        try:
            detail = fetch_advisory_pick_detail(detail_symbol, timeout=20.0)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Pick detail unavailable for {detail_symbol}: {classify_issue(str(exc))}: {exc}")
        else:
            detail_cols = st.columns(5)
            detail_cols[0].metric("Symbol", detail.get("symbol", detail_symbol))
            detail_cols[1].metric("Pred_A", f"{float(detail.get('pred_a') or 0) * 100:.2f}%")
            detail_cols[2].metric("Risk", f"{float(detail.get('risk_prob') or 0) * 100:.2f}%")
            detail_cols[3].metric("FinalScore", f"{float(detail.get('final_score') or 0):.4f}")
            detail_cols[4].metric("Ready", "Yes" if detail.get("ready") else "No")
            warnings = detail.get("warnings") or []
            if warnings:
                st.warning("; ".join(str(item) for item in warnings))
            with st.expander("Raw pick detail"):
                st.json(detail)
    else:
        st.info("No filtered picks available for detail.")

    st.subheader("Ranking Table")
    display = filtered_picks.copy()
    if coverage_map:
        display["Coverage"] = display["Ticker"].map(
            lambda value: "Ready" if (coverage_map.get(normalize_symbol(str(value))) or {}).get("ready") else "Not ready"
        )
        display["Coverage Warnings"] = display["Ticker"].map(
            lambda value: describe_coverage_warning(coverage_map.get(normalize_symbol(str(value))))
        )
    display["Entry_Price"] = display["Entry_Price"].map(lambda value: f"${value:,.2f}")
    display["Pred_A"] = display["Pred_A"].map(lambda value: f"{value * 100:.2f}%")
    display["Risk_Prob_%"] = display["Risk_Prob_%"].map(lambda value: f"{value:.2f}%")
    display["FinalScore"] = display["FinalScore"].map(lambda value: f"{value:.4f}")
    st.dataframe(display, use_container_width=True, hide_index=True)
else:
    _render_pick_diagnostic(picks_warnings, picks_error)


@st.cache_data(show_spinner=False)
def _run_cached_backtest(
    test_start: str,
    end_date: str,
    symbols: tuple[str, ...],
    min_pred_a: float,
    max_risk_prob: float,
    stop_loss: float,
    horizon_days: int,
    cache_version: str,
):
    return run_strategy_backtest(
        test_start=test_start,
        end_date=end_date,
        symbols=list(symbols),
        min_pred_a=min_pred_a,
        max_risk_prob=max_risk_prob,
        stop_loss=stop_loss,
        horizon_days=horizon_days,
    )


st.divider()
st.subheader("Strategy Backtest")

with st.form("strategy_backtest_form"):
    col1, col2, col3, col4 = st.columns(4)
    test_start = col1.date_input("Start date", value=date(2024, 1, 1))
    end_date = col2.date_input("End date", value=date(2026, 5, 29))
    min_pred_a = col3.number_input("Min Pred_A", min_value=0.0, max_value=1.0, value=float(MIN_PRED_A), step=0.01)
    max_risk_prob = col4.number_input(
        "Max Risk_Prob",
        min_value=0.0,
        max_value=1.0,
        value=float(MAX_RISK_PROB_PCT / 100),
        step=0.01,
    )

    col5, col6 = st.columns(2)
    stop_loss = col5.number_input("Stop loss", min_value=0.0, max_value=0.5, value=0.06, step=0.01)
    horizon_days = col6.number_input("Holding window", min_value=1, max_value=60, value=14, step=1)
    symbol_text = st.text_area("Universe", value=", ".join(DEFAULT_SYMBOLS), height=90)
    run_backtest = st.form_submit_button("Run Backtest")

if run_backtest:
    selected_symbols = tuple(symbol.strip().upper().replace(".", "-") for symbol in symbol_text.split(",") if symbol.strip())
    with st.spinner("Running historical inference and backtest..."):
        try:
            result = _run_cached_backtest(
                test_start=str(test_start),
                end_date=str(end_date),
                symbols=selected_symbols,
                min_pred_a=float(min_pred_a),
                max_risk_prob=float(max_risk_prob),
                stop_loss=float(stop_loss),
                horizon_days=int(horizon_days),
                cache_version=BACKTEST_CACHE_VERSION,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Backtest failed: {exc}")
            result = None

    if result is not None:
        metrics = result.metrics
        strategy_total_return = metrics.get("strategy_total_return")
        if strategy_total_return is None and not result.equity_curve.empty:
            strategy_total_return = float(result.equity_curve["Strategy"].iloc[-1] * 100)
        if strategy_total_return is None:
            strategy_total_return = float("nan")

        metric_cols = st.columns(5)
        metric_cols[0].metric("Trades", f"{metrics['trade_count']:,}")
        metric_cols[1].metric("Win Rate", "n/a" if metrics["win_rate"] != metrics["win_rate"] else f"{metrics['win_rate']:.2f}%")
        metric_cols[2].metric(
            "Avg Trade",
            "n/a" if metrics["avg_return_per_trade"] != metrics["avg_return_per_trade"] else f"{metrics['avg_return_per_trade']:.2f}%",
        )
        metric_cols[3].metric(
            "AI Stock Pick CR",
            "n/a" if strategy_total_return != strategy_total_return else f"{strategy_total_return:.2f}%",
        )
        metric_cols[4].metric(
            "SPY Buy & Hold",
            "n/a" if metrics["spy_total_return"] != metrics["spy_total_return"] else f"{metrics['spy_total_return']:.2f}%",
        )

        fig = build_equity_figure(result.equity_curve, min_pred_a=float(min_pred_a), max_risk_prob=float(max_risk_prob))
        st.pyplot(fig, clear_figure=True)

        if not result.exit_reason_counts.empty:
            st.bar_chart(result.exit_reason_counts)

        st.subheader("Executed Trades")
        st.dataframe(result.trades, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Trades CSV",
            data=result.trades.to_csv(index=False).encode("utf-8"),
            file_name="filtered_qualified_trades.csv",
            mime="text/csv",
            disabled=result.trades.empty,
        )
