from __future__ import annotations

import streamlit as st

from data.mock_data import INITIAL_CHAT_MESSAGES


st.set_page_config(page_title="AI Chat", page_icon="💬", layout="wide")

st.title("💬 AI Market Chat")
st.caption("Mock assistant. Backend disabled by default.")

with st.sidebar:
    st.header("Context")
    symbols = st.text_input("Symbols", "NVDA, MSFT, LLY")
    horizon = st.selectbox("Horizon", ["Intraday", "1-4 weeks", "1-3 months", "6-12 months"], index=1)
    risk = st.select_slider("Risk tolerance", ["Low", "Medium", "High"], value="Medium")
    st.markdown("<span style='color:#67e8f9;font-weight:700'>Mock mode active</span>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = INITIAL_CHAT_MESSAGES.copy()

prompt_cols = st.columns(3)
sample_prompts = [
    "Summarize today's market regime",
    "Compare NVDA and MSFT",
    "Show risks in current picks",
]
for col, sample in zip(prompt_cols, sample_prompts):
    if col.button(sample, width="stretch"):
        st.session_state.messages.append({"role": "user", "content": sample})
        st.session_state.messages.append({"role": "assistant", "content": f"Mock view for {symbols}: {sample}. Horizon {horizon}, risk {risk}. Signals favor quality growth with controlled position sizing."})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask ORCA about markets or stocks..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    reply = f"Mock ORCA reply: For {symbols}, {horizon.lower()} setup looks constructive but risk is {risk.lower()}. Watch breadth, revisions, and volatility before adding exposure."
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
