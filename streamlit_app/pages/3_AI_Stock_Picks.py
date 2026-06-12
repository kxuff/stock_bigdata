from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from components.styles import inject_global_styles
from services.backtest_api import DEFAULT_SYMBOLS, build_equity_figure, run_strategy_backtest
from services.ml_inference_api import (
    MAX_RISK_PROB_PCT,
    MIN_PRED_A,
    ensure_latest_ml_inference,
    normalize_ml_inference_picks,
)


st.set_page_config(page_title="AI Stock Picks", page_icon="⭐", layout="wide")
inject_global_styles()
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
st.caption("Ranked model signals from the latest EOD inference batch.")

with st.expander("Output contract"):
    st.markdown(
        f"""
        - `Date`: ngày phát sinh tín hiệu mua.
        - `Ticker`: mã cổ phiếu.
        - `Entry_Price`: giá mua đề xuất tại thời điểm tín hiệu.
        - `Pred_A`: mức tăng giá dự báo bởi mô hình, dạng thập phân.
        - `Risk_Prob_%`: chỉ số rủi ro dự báo của mô hình, hiển thị theo phần trăm.
        - `FinalScore`: điểm xếp hạng cuối cùng, tương đương `Pred_A * (1 - RiskProb)`.
        - Chỉ hiển thị mã có `Pred_A >= {MIN_PRED_A:.2f}` và `Risk_Prob <= {MAX_RISK_PROB_PCT / 100:.2f}`.
        """
    )

with st.spinner("Checking latest EOD signal batch..."):
    try:
        availability = ensure_latest_ml_inference()
        if availability.prediction_path is None:
            picks = None
        else:
            picks = normalize_ml_inference_picks(pd.read_parquet(availability.prediction_path), limit=100)
    except Exception as exc:
        availability = None
        st.warning(f"Cannot load ML inference output: {exc}")
        picks = None

if availability is not None:
    status = f"Expected latest signal date: `{availability.expected_signal_date.isoformat()}`."
    if availability.prediction_path is not None:
        status += f" Reading `{availability.prediction_path}`."
    if availability.refreshed:
        st.success(status + " Refreshed automatically.")
    elif availability.refresh_error:
        st.warning(status + f" Auto-refresh failed: {availability.refresh_error}")
    else:
        st.caption(status)

if picks is not None and not picks.empty:
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

    top = filtered_picks.head(3)
    metrics = st.columns(3)
    metrics[0].metric("Signals", len(filtered_picks))
    metrics[1].metric("Best FinalScore", f"{filtered_picks['FinalScore'].max():.4f}" if not filtered_picks.empty else "n/a")
    metrics[2].metric("Average Pred_A", f"{filtered_picks['Pred_A'].mean() * 100:.2f}%" if not filtered_picks.empty else "n/a")

    for _, pick in top.iterrows():
        badge_class = "buy" if pick["Pred_A"] > 0 and pick["FinalScore"] > 0 else "watch"
        st.markdown(
            f"""
            <div class='pick-card'>
              <span class='badge {badge_class}'>MODEL PICK</span>
              <h3>{pick['Ticker']}</h3>
              <p>
                <b>Entry:</b> ${pick['Entry_Price']:.2f} ·
                <b>Pred_A:</b> {pick['Pred_A'] * 100:.2f}% ·
                <b>Risk:</b> {pick['Risk_Prob_%']:.2f}% ·
                <b>FinalScore:</b> {pick['FinalScore']:.4f}
              </p>
              <p class='muted'>Signal date: {pick['Date']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Ranking Table")
    display = filtered_picks.copy()
    display["Entry_Price"] = display["Entry_Price"].map(lambda value: f"${value:,.2f}")
    display["Pred_A"] = display["Pred_A"].map(lambda value: f"{value * 100:.2f}%")
    display["Risk_Prob_%"] = display["Risk_Prob_%"].map(lambda value: f"{value:.2f}%")
    display["FinalScore"] = display["FinalScore"].map(lambda value: f"{value:.4f}")
    st.dataframe(display, width="stretch", hide_index=True)
else:
    st.info(
        "No qualified ML picks found. The page only shows signals with "
        f"Pred_A >= {MIN_PRED_A * 100:.0f}% and Risk_Prob <= {MAX_RISK_PROB_PCT:.0f}%."
    )


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
        except Exception as exc:
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
        st.dataframe(result.trades, width="stretch", hide_index=True)
        st.download_button(
            "Download Trades CSV",
            data=result.trades.to_csv(index=False).encode("utf-8"),
            file_name="filtered_qualified_trades.csv",
            mime="text/csv",
            disabled=result.trades.empty,
        )
