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

    items = []
    for _, row in df.iterrows():
        change = row.get("change_value")
        pct = row.get("change_pct")
        positive = pd.notna(change) and change > 0
        negative = pd.notna(change) and change < 0
        css_class = "up" if positive else "down" if negative else "flat"
        arrow = "&#9650;" if positive else "&#9660;" if negative else "&#9632;"
        indicator = str(row.get("indicator", "N/A"))
        market_badge = "VN" if any(token in indicator.upper() for token in ["VN", "HNX", "HANG"]) else "US"
        items.append(
            "<div class='ticker-item'>"
            f"<div class='ticker-name'><span class='ticker-flag'>{market_badge}</span>{escape(indicator)}</div>"
            f"<div class='ticker-price {css_class}'>{_fmt_number(row.get('close'))}</div>"
            f"<div class='ticker-change {css_class}'>{arrow} {_fmt_number(change)} ({_fmt_number(pct)}%)</div>"
            "</div>"
        )

    st.markdown(
        f"<div class='ticker-strip'>{''.join(items)}<div class='ticker-arrow'>&rsaquo;</div></div>",
        unsafe_allow_html=True,
    )
