from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st


def render_news_cards(df: pd.DataFrame, page: int, page_size: int = 8) -> None:
    if df.empty:
        st.info("No news matched the current filters.")
        return

    start = (page - 1) * page_size
    for _, row in df.iloc[start : start + page_size].iterrows():
        image = row.get("image")
        if not image or pd.isna(image):
            image = "https://placehold.co/256x168/111827/94a3b8?text=Market"
        headline = escape(str(row.get("headline") or "Untitled"))
        source = escape(str(row.get("source") or "Unknown source"))
        symbol = escape(str(row.get("symbol") or ""))
        event_time = row.get("event_timestamp") or row.get("datetime")
        timestamp = event_time.strftime("%Y-%m-%d %H:%M") if hasattr(event_time, "strftime") else escape(str(event_time or ""))
        url = escape(str(row.get("url") or "#"))
        st.markdown(
            f"""
            <div class="news-card">
                <img src="{escape(str(image))}" alt="news image">
                <div class="news-body">
                    <div class="news-title">{headline}</div>
                    <div class="news-meta">{symbol} - {source} - {timestamp}</div>
                    <a class="news-link" href="{url}" target="_blank" rel="noopener noreferrer">Open article</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
