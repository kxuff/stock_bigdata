"""ORCA AI Chat — main page (slim orchestrator)."""
from __future__ import annotations

import streamlit as st

from chat import components as C
from chat import jobs as J
from chat import state as S
from chat.styles import inject as inject_css

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ORCA · AI Chat", page_icon="", layout="wide")
inject_css()
S.init()

# ── Backend status (lazy init) ────────────────────────────────────────────────
if st.session_state.orca_backend_status is None:
    st.session_state.orca_backend_status = J.check_backend()

backend      = st.session_state.orca_backend_status
backend_state = backend.get("state", "Offline")
api_offline   = backend_state == "Offline"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="orca-kicker">ORCA Context</div>', unsafe_allow_html=True)
    risk = st.select_slider("Risk tolerance", ["Low", "Medium", "High"], value="Medium")
    st.caption("Symbols and portfolio details should be written directly in your prompt. ORCA uses the model's 14-day horizon internally.")

    st.markdown("---")
    st.markdown('<div class="orca-kicker">Backend</div>', unsafe_allow_html=True)
    C.render_backend_pill(backend_state, backend.get("error"))

    if st.button("Refresh connection", width="stretch"):
        st.session_state.orca_backend_status = J.check_backend()
        st.rerun()

    st.markdown("---")
    st.caption("All advisory jobs route through the live ORCA API. No trades executed.")

# ── Page header ───────────────────────────────────────────────────────────────
col_title, col_clear = st.columns([1, 0.18], vertical_alignment="center")
col_title.title("AI Chat")
col_title.caption("Ask ORCA about markets, symbols, portfolios, or data health.")
if col_clear.button("Clear", width="stretch", help="Clear conversation"):
    S.clear_chat()
    st.rerun()

if api_offline:
    st.error("**ORCA API offline.** Start the backend before submitting advisory jobs.")

# ── Quick-action chips ────────────────────────────────────────────────────────
st.markdown('<div class="orca-kicker">Quick questions</div>', unsafe_allow_html=True)

QUICK = [
    ("Market brief",        "Give me a quick market brief"),
    ("Top stocks",          "Show me the top stocks to watch right now"),
    ("Stock advice",        "Should I buy GILD? Explain the risk and what would change your view."),
    ("Compare",             "Compare AAPL vs MSFT vs AMZN"),
    ("Rebalance portfolio", "Rebalance my portfolio: AAPL 35%, MSFT 30%, AMZN 20%, GOOGL 15%. Moderate risk."),
]
chip_cols = st.columns(len(QUICK))
for col, (label, prompt) in zip(chip_cols, QUICK):
    with col:
        if st.button(label, width="stretch", disabled=api_offline, key=f"chip-{label}"):
            S.add_user(prompt)
            with st.spinner("Routing to ORCA…"):
                reply = J.submit(prompt, risk)
            if reply:
                S.add_assistant(reply)
            st.rerun()

# ── Pending jobs tracker ──────────────────────────────────────────────────────
@st.fragment(run_every="2s")
def render_active_jobs() -> None:
    jobs = S.pending_jobs()
    if not jobs:
        return

    C.kicker("Active ORCA jobs")
    should_refresh_page = False
    for job in list(jobs):
        status = S.display_status(job)
        jcols  = st.columns([0.9, 0.7, 2.2, 1.2, 0.7, 0.7], vertical_alignment="center")
        jcols[0].markdown("**ORCA**")
        jcols[1].markdown(f"`{status}`")
        jcols[2].caption(C._truncate(job.get("prompt"), 60))
        duration_slot = jcols[3].empty()
        duration_slot.caption(S.fmt_duration(job))
        if status in {"failed", "stale"} and jcols[4].button("Retry", key=f"retry-{job['job_id']}", help="Retry"):
            J.retry_job(job, risk)
            st.rerun()
        if jcols[5].button("Remove", key=f"rm-{job['job_id']}", help="Remove"):
            S.remove_job(job["job_id"])
            st.rerun()
        if status in {"queued", "running"} and not job.get("events_complete"):
            was_done = job.get("result_fetched") or job.get("events_complete")
            J.poll_job(job)
            is_done = job.get("result_fetched") or job.get("events_complete")
            should_refresh_page = should_refresh_page or (not was_done and is_done)

    if should_refresh_page:
        st.rerun()


render_active_jobs()

# ── Retry banner ──────────────────────────────────────────────────────────────
if st.session_state.submit_retry:
    retry_prompt = st.session_state.submit_retry
    c1, c2 = st.columns([1, 0.22])
    c1.warning(f"Last submit failed. Retry: *{C._truncate(retry_prompt, 60)}*")
    if c2.button("Retry", disabled=api_offline):
        S.add_user(retry_prompt)
        reply = J.submit(retry_prompt, risk)
        if reply:
            S.add_assistant(reply)
        st.session_state.submit_retry = None
        st.rerun()

S.sync_jobs_to_query()

# ── Conversation ──────────────────────────────────────────────────────────────
C.kicker("Conversation")
messages = st.session_state.messages

if not messages:
    C.render_chat_empty()
else:
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "decision":
                C.render_decision(msg.get("decision") or {})
            elif msg.get("type") == "agent_response":
                C.render_agent_response(msg.get("response") or {})
            else:
                st.markdown(msg.get("content", ""))

# ── Chat input ────────────────────────────────────────────────────────────────
if user_prompt := st.chat_input("Ask ORCA about a symbol, market topic, or portfolio…", disabled=api_offline):
    S.add_user(user_prompt)
    with st.chat_message("user"):
        st.markdown(user_prompt)

    reply = None
    if api_offline:
        reply = "ORCA API offline. Start the backend first."
    else:
        with st.spinner("Routing to ORCA…"):
            reply = J.submit(user_prompt, risk)
        # Flag retry if error
        if reply and any(k in reply for k in ("**Api", "**Timeout", "**Malformed")):
            st.session_state.submit_retry = user_prompt

    if reply:
        S.add_assistant(reply)
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        st.rerun()
