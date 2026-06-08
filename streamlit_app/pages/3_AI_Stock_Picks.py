from __future__ import annotations

from datetime import date
import os

import pandas as pd
import streamlit as st

from services.backtest_api import DEFAULT_SYMBOLS, build_equity_figure, run_strategy_backtest
from services.ml_inference_api import (
    MAX_RISK_PROB_PCT,
    MIN_PRED_A,
    ensure_latest_ml_inference,
    normalize_ml_inference_picks,
)


st.set_page_config(page_title="AI Stock Picks", page_icon="⭐", layout="wide")
BACKTEST_CACHE_VERSION = "cr_metric_v2"

st.markdown("""
<style>
.pick-card{background:#111827;border:1px solid #374151;border-radius:8px;padding:16px;margin-bottom:10px}
.badge{display:inline-block;padding:3px 8px;border-radius:6px;font-weight:800;font-size:12px}
.buy{background:#064e3b;color:#6ee7b7}.watch{background:#78350f;color:#fcd34d}.risk{background:#7f1d1d;color:#fecaca}
.muted{color:#9ca3af}
</style>
""", unsafe_allow_html=True)

st.title("⭐ AI Stock Picks")
st.caption("Backtest strategy performance, then generate recommendations for a selected signal date.")


def _format_recommendations(picks: pd.DataFrame) -> pd.DataFrame:
    display = picks.copy()
    display["Entry_Price"] = display["Entry_Price"].map(lambda value: f"${value:,.2f}")
    display["Pred_A"] = display["Pred_A"].map(lambda value: f"{value * 100:.2f}%" if value <= 1 else f"{value:.2f}%")
    display["Risk_Prob_%"] = display["Risk_Prob_%"].map(lambda value: f"{value:.2f}%")
    display["FinalScore"] = display["FinalScore"].map(lambda value: f"{value * 100:.2f}%" if value <= 1 else f"{value:.2f}%")
    return display


def _render_recommendations(picks: pd.DataFrame, selected_date: str) -> None:
    filtered_picks = picks[picks["Date"] == selected_date].sort_values("FinalScore", ascending=False)
    if filtered_picks.empty:
        st.info(
            "No qualified ML picks found for this date. The page only shows signals with "
            f"Pred_A >= {MIN_PRED_A * 100:.0f}% and Risk_Prob <= {MAX_RISK_PROB_PCT:.0f}%."
        )
        return

    metrics = st.columns(3)
    metrics[0].metric("Signals", len(filtered_picks))
    metrics[1].metric("Best FinalScore", f"{filtered_picks['FinalScore'].max() * 100:.2f}%")
    metrics[2].metric("Average Pred_A", f"{filtered_picks['Pred_A'].mean() * 100:.2f}%")

    for _, pick in filtered_picks.head(3).iterrows():
        pred_a_pct = pick["Pred_A"] * 100 if pick["Pred_A"] <= 1 else pick["Pred_A"]
        final_score_pct = pick["FinalScore"] * 100 if pick["FinalScore"] <= 1 else pick["FinalScore"]
        st.markdown(
            f"""
            <div class='pick-card'>
              <span class='badge buy'>MODEL PICK</span>
              <h3>{pick['Ticker']}</h3>
              <p>
                <b>Entry:</b> ${pick['Entry_Price']:.2f} ·
                <b>Pred_A:</b> {pred_a_pct:.2f}% ·
                <b>Risk:</b> {pick['Risk_Prob_%']:.2f}% ·
                <b>FinalScore:</b> {final_score_pct:.2f}%
              </p>
              <p class='muted'>Signal date: {pick['Date']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.dataframe(_format_recommendations(filtered_picks), width="stretch", hide_index=True)


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
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            result = None
            st.session_state.pop("strategy_backtest_result", None)

        if result is not None:
            st.session_state.strategy_backtest_result = result
            st.session_state.strategy_backtest_params = {
                "min_pred_a": float(min_pred_a),
                "max_risk_prob": float(max_risk_prob),
            }

result = st.session_state.get("strategy_backtest_result")
result_params = st.session_state.get(
    "strategy_backtest_params",
    {"min_pred_a": float(min_pred_a), "max_risk_prob": float(max_risk_prob)},
)

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

    fig = build_equity_figure(
        result.equity_curve,
        min_pred_a=float(result_params["min_pred_a"]),
        max_risk_prob=float(result_params["max_risk_prob"]),
    )
    st.pyplot(fig, clear_figure=True)

    if not result.exit_reason_counts.empty:
        st.bar_chart(result.exit_reason_counts)

    st.subheader("Executed Trades")
    st.dataframe(result.trades, width="stretch", hide_index=True)
    st.download_button(
        "Download Trades CSV",
        data=result.trades.to_csv(index=False).encode("utf-8"),
        file_name="filtered_qualified_trades.csv",
        mime="text/csv",
        disabled=result.trades.empty,
    )

st.divider()
st.subheader("Recommendations By Date")

with st.form("recommendation_form"):
    rec_col1, rec_col2 = st.columns([1, 3])
    recommendation_date = rec_col1.date_input("Signal date", value=date.today())
    get_recommendation = rec_col2.form_submit_button("Get Recommendation")

if get_recommendation:
    os.environ["ML_INFERENCE_AUTO_REFRESH"] = "true"
    with st.spinner("Generating recommendations..."):
        try:
            availability = ensure_latest_ml_inference(today=recommendation_date)
            if availability.prediction_path is None:
                st.session_state.recommendation_status = (
                    f"No prediction file was created for {availability.expected_signal_date.isoformat()}."
                )
                st.session_state.recommendation_picks = pd.DataFrame()
                st.session_state.recommendation_date = availability.expected_signal_date.isoformat()
            else:
                picks = normalize_ml_inference_picks(pd.read_parquet(availability.prediction_path), limit=100)
                st.session_state.recommendation_picks = picks
                st.session_state.recommendation_date = availability.expected_signal_date.isoformat()
                status = f"Signal date: `{availability.expected_signal_date.isoformat()}`."
                if availability.prediction_path is not None:
                    status += f" Reading `{availability.prediction_path}`."
                if availability.refresh_error:
                    status += f" Refresh warning: {availability.refresh_error}"
                st.session_state.recommendation_status = status
        except Exception as exc:
            st.session_state.recommendation_picks = pd.DataFrame()
            st.session_state.recommendation_date = recommendation_date.isoformat()
            st.session_state.recommendation_status = f"Cannot load ML recommendations: {exc}"

recommendation_status = st.session_state.get("recommendation_status")
recommendation_picks = st.session_state.get("recommendation_picks")
recommendation_date_value = st.session_state.get("recommendation_date")

if recommendation_status:
    st.caption(recommendation_status)

if isinstance(recommendation_picks, pd.DataFrame) and recommendation_date_value:
    _render_recommendations(recommendation_picks, recommendation_date_value)
