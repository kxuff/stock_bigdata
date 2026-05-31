from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st


def _fmt_number(value: float | int | None) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):,.2f}"


def render_market_overview(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No market indicator rows found.")
        return

    cols = st.columns(min(4, len(df)))
    for idx, (_, row) in enumerate(df.iterrows()):
        change = row.get("change_value")
        pct = row.get("change_pct")
        positive = pd.notna(change) and change > 0
        negative = pd.notna(change) and change < 0
        css_class = "up" if positive else "down" if negative else "flat"
        arrow = "&uarr;" if positive else "&darr;" if negative else "&minus;"
        latest_time = row.get("latest_time")
        timestamp = latest_time.strftime("%Y-%m-%d %H:%M") if hasattr(latest_time, "strftime") else str(latest_time)
        cols[idx % len(cols)].markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-name">{escape(str(row.get("indicator", "N/A")))}</div>
                <div class="kpi-price">{_fmt_number(row.get("close"))}</div>
                <div class="kpi-change {css_class}">{arrow} {_fmt_number(change)} ({_fmt_number(pct)}%)</div>
                <div class="kpi-time">Last tick: {escape(timestamp)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
