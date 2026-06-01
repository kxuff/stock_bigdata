from __future__ import annotations

from datetime import datetime
from math import ceil

import pandas as pd
import streamlit as st

from components.market import render_market_overview
from components.news import render_news_cards
from components.styles import inject_global_styles
from data_loader import (
    load_alert_filter_values,
    load_alerts,
    load_latest_market,
    load_market_overview,
    load_news,
    load_symbols,
)


st.set_page_config(page_title="Market Dashboard", page_icon="chart", layout="wide")
inject_global_styles()


def clear_caches() -> None:
    st.cache_data.clear()
    st.session_state["last_manual_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def page_slice(df: pd.DataFrame, page: int, page_size: int) -> pd.DataFrame:
    start = (page - 1) * page_size
    return df.iloc[start : start + page_size]


def format_market_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in ["open", "high", "low", "close"]:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:,.2f}")
    display["volume"] = display["volume"].map(lambda value: "" if pd.isna(value) else f"{value:,.0f}")
    display["day_change_pct"] = display["day_change_pct"].map(lambda value: "" if pd.isna(value) else f"{value:+.2f}%")
    return display.rename(
        columns={
            "symbol": "Symbol",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "day_change_pct": "Day Change %",
            "datetime": "Last Tick",
        }
    )


st.markdown(
    """
    <div class="market-header">
      <h1>Market Dashboard</h1>
      <p>Live KPI tape, news feed, stock monitor, and alert center from PostgreSQL.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    st.caption("Market, stock monitor, and alerts update every 60 seconds without reloading the full page.")
    st.caption("News updates every 5 minutes.")
    if st.button("Refresh now", use_container_width=True):
        clear_caches()
        st.rerun()
    st.caption(f"Last manual refresh: {st.session_state.get('last_manual_refresh', 'not used')}")
    st.caption(f"Page generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    symbols = load_symbols()
except Exception as exc:
    st.error(f"Cannot connect to PostgreSQL or load symbols: {exc}")
    st.stop()


@st.fragment(run_every="60s")
def render_market_fragment() -> None:
    st.markdown('<div class="section-title">Market Overview</div>', unsafe_allow_html=True)
    try:
        overview = load_market_overview(limit=50)
        render_market_overview(overview)
    except Exception as exc:
        st.warning(f"Cannot load market indicators: {exc}")


@st.fragment(run_every="60s")
def render_stock_monitor_fragment() -> None:
    st.markdown('<div class="section-title">Stock Monitor</div>', unsafe_allow_html=True)
    search = st.text_input("Search symbol", "")
    selected_universe = [symbol for symbol in symbols if search.upper() in symbol.upper()] if search else symbols
    filter_col, sort_col_box, direction_col, size_col = st.columns([2, 1, 1, 1])
    quick_symbols = filter_col.multiselect("Filter symbols", selected_universe[:500], default=[])
    sort_col = sort_col_box.selectbox("Sort by", ["symbol", "close", "volume", "day_change_pct"], index=0)
    sort_desc = direction_col.toggle("Descending", value=False)
    page_size = size_col.selectbox("Rows per page", [10, 25, 50, 100], index=1)

    market_df = load_latest_market(tuple(quick_symbols) if quick_symbols else None, limit=1000)
    if search:
        market_df = market_df[market_df["symbol"].str.contains(search, case=False, na=False)]
    if sort_col in market_df.columns:
        market_df = market_df.sort_values(sort_col, ascending=not sort_desc, na_position="last")

    total_pages = max(1, ceil(len(market_df) / page_size))
    market_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    st.caption(f"{len(market_df):,} symbols - page {market_page}/{total_pages}")
    st.dataframe(format_market_table(page_slice(market_df, int(market_page), int(page_size))), width="stretch", hide_index=True)


@st.fragment(run_every="300s")
def render_news_fragment() -> None:
    st.markdown('<div class="section-title">News Feed</div>', unsafe_allow_html=True)
    news_filter_col, news_body_col = st.columns([0.9, 2.1])
    with news_filter_col:
        news_symbol = st.selectbox("News symbol", ["All"] + symbols, index=0)
        headline_search = st.text_input("Search headline")
        news_page_size = st.selectbox("News per page", [5, 8, 12, 20], index=1)
    with news_body_col:
        news_df = load_news(news_symbol, headline_search, limit=300)
        news_total_pages = max(1, ceil(len(news_df) / news_page_size))
        page_col, count_col = st.columns([1, 3])
        news_page = page_col.number_input("News page", min_value=1, max_value=news_total_pages, value=1, step=1)
        count_col.markdown(
            f"<div class='status-line'>{len(news_df):,} articles - cache TTL 5 minutes</div>",
            unsafe_allow_html=True,
        )
        render_news_cards(news_df, int(news_page), int(news_page_size))


@st.fragment(run_every="60s")
def render_alerts_fragment() -> None:
    st.markdown('<div class="section-title">Alerts Center</div>', unsafe_allow_html=True)
    try:
        alert_symbols, alert_types, alert_levels = load_alert_filter_values()
    except Exception:
        alert_symbols, alert_types, alert_levels = [], [], []

    alert_col1, alert_col2, alert_col3, alert_col4 = st.columns(4)
    selected_alert_symbols = alert_col1.multiselect("Alert symbols", alert_symbols, default=[])
    selected_alert_types = alert_col2.multiselect("Alert types", alert_types, default=[])
    selected_alert_levels = alert_col3.multiselect("Alert levels", alert_levels, default=[])
    date_range = alert_col4.date_input("Event time range", value=())
    start_date = date_range[0] if isinstance(date_range, tuple) and len(date_range) >= 1 else None
    end_date = date_range[1] if isinstance(date_range, tuple) and len(date_range) >= 2 else None

    alerts = load_alerts(
        tuple(selected_alert_symbols),
        tuple(selected_alert_types),
        tuple(selected_alert_levels),
        start_date,
        end_date,
        limit=1000,
    )
    if alerts.empty:
        st.info("No alerts matched the current filters.")
        return

    def level_style(row: pd.Series) -> list[str]:
        level = str(row.get("Alert Level", "")).lower()
        color = ""
        if level == "critical":
            color = "background-color: rgba(239, 68, 68, .16)"
        elif level == "warning":
            color = "background-color: rgba(245, 158, 11, .16)"
        elif level == "info":
            color = "background-color: rgba(56, 189, 248, .14)"
        return [color] * len(row)

    display_alerts = alerts.rename(
        columns={
            "symbol": "Symbol",
            "event_time": "Event Time",
            "alert_type": "Alert Type",
            "alert_level": "Alert Level",
            "message": "Message",
        }
    )[["Symbol", "Event Time", "Alert Type", "Alert Level", "Message"]]
    st.dataframe(display_alerts.style.apply(level_style, axis=1), width="stretch", hide_index=True)


render_market_fragment()
render_stock_monitor_fragment()
render_news_fragment()
render_alerts_fragment()
