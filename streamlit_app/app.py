from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="ORCA AI Stock Desk",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {padding: 2rem; border-radius: 1.25rem; background: linear-gradient(135deg, #052e2b, #0f172a 55%, #111827); border: 1px solid #164e63;}
    .badge {display: inline-block; padding: .25rem .6rem; border-radius: 999px; font-weight: 700; font-size: .8rem;}
    .emerald {background:#064e3b; color:#6ee7b7;} .cyan {background:#164e63; color:#67e8f9;}
    .amber {background:#78350f; color:#fcd34d;} .rose {background:#881337; color:#fda4af;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero">', unsafe_allow_html=True)
st.title("ORCA AI Stock Desk")
st.subheader("Mock-first Streamlit cockpit for dashboard, chat, and AI stock picks.")
st.markdown("<span class='badge emerald'>Mock Data</span> <span class='badge cyan'>No backend calls by default</span>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.info("Use sidebar pages: Dashboard, AI Chat, AI Stock Picks.")
