from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st


COLLAPSED_ITEMS = 6

INDICATOR_NAMES = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "CL=F": "WTI Crude Oil",
    "GC=F": "Gold",
    "SI=F": "Silver",
    "NG=F": "Natural Gas",
    "^DJI": "Dow Jones",
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
    "^RUT": "Russell 2000",
    "^VIX": "VIX",
    "VNINDEX": "VN-Index",
    "HNXINDEX": "HNX-Index",
    "UPCOMINDEX": "UPCoM-Index",
}


def _fmt_number(value: float | int | None) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):,.2f}"


def _indicator_name(value: object) -> str:
    indicator = str(value or "N/A").strip()
    if not indicator or indicator.lower() == "nan":
        return "N/A"
    if indicator in INDICATOR_NAMES:
        return INDICATOR_NAMES[indicator]

    clean = indicator.lstrip("^").replace("-USD", "").replace("=F", "")
    return clean.replace("_", " ").replace("-", " ").title()


def render_market_overview(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No market indicator rows found.")
        return

    expanded = st.session_state.get("market_overview_expanded", False)
    visible_df = df if expanded else df.head(COLLAPSED_ITEMS)

    items = []
    for _, row in visible_df.iterrows():
        change = row.get("change_value")
        pct = row.get("change_pct")
        positive = pd.notna(change) and change > 0
        negative = pd.notna(change) and change < 0
        css_class = "up" if positive else "down" if negative else "flat"
        arrow = "&#9650;" if positive else "&#9660;" if negative else "&#9632;"
        indicator = _indicator_name(row.get("indicator", "N/A"))
        items.append(
            "<div class='ticker-item'>"
            f"<div class='ticker-name'>{escape(indicator)}</div>"
            f"<div class='ticker-price {css_class}'>{_fmt_number(row.get('close'))}</div>"
            f"<div class='ticker-change {css_class}'>{arrow} {_fmt_number(change)} ({_fmt_number(pct)}%)</div>"
            "</div>"
        )

    st.markdown(
        f"<div class='ticker-strip'>{''.join(items)}</div>",
        unsafe_allow_html=True,
    )

    if len(df) > COLLAPSED_ITEMS:
        label = "Thu gon" if expanded else f"Mo rong ({len(df) - COLLAPSED_ITEMS} chi so)"
        if st.button(label, key="market_overview_toggle"):
            st.session_state["market_overview_expanded"] = not expanded
            st.rerun()
