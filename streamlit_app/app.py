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
      <p>Local ORCA demo console for market data, data health, AI chat, and model-ranked picks.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(4)
with cols[0]:
    st.page_link("pages/1_Dashboard.py", label="Dashboard")
    st.caption("Market overview, monitor, alerts, and symbol chart.")
with cols[1]:
    st.page_link("pages/4_Data_Health.py", label="Data Health")
    st.caption("ORCA API status, coverage, and pick diagnostics.")
with cols[2]:
    st.page_link("pages/2_AI_Chat.py", label="AI Chat")
    st.caption("Agent jobs, routed market queries, and advisory decisions.")
with cols[3]:
    st.page_link("pages/3_AI_Stock_Picks.py", label="AI Stock Picks")
    st.caption("Ranked EOD model signals and ORCA job handoff.")

st.divider()
st.info("Demo defaults: AAPL, MSFT, NVDA with ORCA API at http://127.0.0.1:8000.")
