from __future__ import annotations

import plotly.express as px
import streamlit as st

from data.mock_data import KPI_METRICS, RECENT_DECISIONS, TODAYS_BRIEF, TREND_DATA


st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

st.markdown("""
<style>.brief{background:#0f172a;border:1px solid #164e63;border-radius:16px;padding:18px}.pill{padding:4px 10px;border-radius:999px;background:#064e3b;color:#6ee7b7;font-weight:700}</style>
""", unsafe_allow_html=True)

st.title("📈 AI Market Dashboard")
st.caption("Placeholder dark stock dashboard powered by reusable mock data.")

cols = st.columns(4)
for col, metric in zip(cols, KPI_METRICS):
    col.metric(metric["label"], metric["value"], metric["delta"])

left, right = st.columns([2, 1])
with left:
    fig = px.line(TREND_DATA, x="date", y=["AI confidence", "Market breadth"], markers=True, template="plotly_dark")
    fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.markdown("<div class='brief'><span class='pill'>Today's brief</span>", unsafe_allow_html=True)
    for item in TODAYS_BRIEF:
        st.markdown(f"- {item}")
    st.markdown("</div>", unsafe_allow_html=True)

st.subheader("Recent AI Decisions")
st.dataframe(RECENT_DECISIONS, use_container_width=True, hide_index=True)
