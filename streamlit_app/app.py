from __future__ import annotations

import streamlit as st

from components.styles import inject_global_styles


st.set_page_config(
    page_title="Real-time Stock Market Desk",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_styles()

st.markdown(
    """
    <div class="market-header">
      <h1>Real-time Stock Market Desk</h1>
      <p>PostgreSQL-backed Streamlit dashboard for market overview, news, stock monitor, and alerts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info("Open the Dashboard page from the sidebar to monitor live market data.")
