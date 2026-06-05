"""ORCA AI Chat main page."""
from __future__ import annotations

import streamlit as st

from chat import components as C
from chat import jobs as J
from chat import state as S
from chat.styles import inject as inject_css
from services.advisory_api import fetch_data_coverage
from services.demo_diagnostics import classify_issue, coverage_warning as describe_coverage_warning, normalize_symbol, should_disable_quick_action


st.set_page_config(page_title="ORCA - AI Chat", page_icon="*", layout="wide")
inject_css()
S.init()

if st.session_state.orca_backend_status is None:
    st.session_state.orca_backend_status = J.check_backend()

backend = st.session_state.orca_backend_status
backend_state = backend.get("state", "Offline")
api_offline = backend_state == "Offline"

with st.sidebar:
    st.markdown('<div class="orca-kicker">ORCA Context</div>', unsafe_allow_html=True)
    symbol_input = st.text_input("Symbol context", "NVDA", placeholder="e.g. AAPL")
    horizon = st.selectbox(
        "Investment horizon",
        ["Intraday", "1-4 weeks", "1-3 months", "6-12 months"],
        index=1,
    )
    risk = st.select_slider("Risk tolerance", ["Low", "Medium", "High"], value="Medium")
    with st.expander("Portfolio context", expanded=False):
        holdings_text = st.text_area(
            "Holdings",
            J.DEFAULT_PORTFOLIO_HOLDINGS,
            help="Use SYMBOL:weight pairs separated by commas.",
        )
        pcols = st.columns(2)
        min_cash_weight = pcols[0].number_input("Min cash %", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
        max_single_asset_weight = pcols[1].number_input("Max asset %", min_value=1.0, max_value=100.0, value=40.0, step=1.0)
        portfolio_metadata, portfolio_warnings = J.build_portfolio_metadata(
            holdings_text,
            min_cash_weight=min_cash_weight,
            max_single_asset_weight=max_single_asset_weight,
        )
        for warning in portfolio_warnings:
            st.warning(warning)
    primary_symbol = normalize_symbol(symbol_input)
    coverage_ready = False
    coverage_warning = None
    coverage_issue = None

    st.markdown("---")
    st.markdown('<div class="orca-kicker">Backend</div>', unsafe_allow_html=True)
    C.render_backend_pill(backend_state, backend.get("error"))
    if not primary_symbol:
        coverage_warning = "Enter a symbol before submitting an ORCA job."
        st.caption("Symbol-specific advisory actions need a symbol.")
    elif not api_offline:
        try:
            coverage = fetch_data_coverage([primary_symbol], timeout=10.0)
            row = (coverage.get("rows") or [{}])[0]
            coverage_ready = bool(row.get("ready"))
            if coverage_ready:
                st.success(f"Data ready for {primary_symbol}.")
            else:
                reason = describe_coverage_warning(row) or "required data is not ready"
                coverage_issue = classify_issue(reason)
                coverage_warning = f"Data not ready for {primary_symbol}: {reason}"
                st.warning(f"{coverage_issue}: {reason}")
        except Exception as exc:  # noqa: BLE001
            coverage_warning = f"Coverage check failed: {exc}"
            coverage_issue = classify_issue(str(exc))
            st.warning(coverage_warning)

    if st.button("Refresh connection", use_container_width=True):
        st.session_state.orca_backend_status = J.check_backend()
        st.rerun()

    st.markdown("---")
    st.page_link("pages/4_Data_Health.py", label="Open Data Health")
    st.caption("All advisory jobs route through the live ORCA API. No trades executed.")

submit_disabled = api_offline

col_title, col_clear = st.columns([1, 0.18], vertical_alignment="center")
col_title.title("AI Chat")
col_title.caption("Ask ORCA about markets, symbols, portfolios, or data health.")
if col_clear.button("Clear", use_container_width=True, help="Clear conversation"):
    S.clear_chat()
    st.rerun()

if api_offline:
    st.error("ORCA API offline. Start the backend before submitting advisory jobs.")
elif coverage_warning:
    st.warning(f"{coverage_issue or 'Data coverage'}: {coverage_warning}")

st.markdown('<div class="orca-kicker">Quick questions</div>', unsafe_allow_html=True)

quick = [
    ("Market brief", "Give me a quick market brief"),
    ("Top stocks", "Show me the top stocks to watch right now"),
    (f"Advise {primary_symbol or 'symbol'}", f"Should I buy {primary_symbol}?"),
    ("Portfolio allocation", "Recommend allocation for my portfolio"),
    ("Compare", f"Compare {primary_symbol} vs AAPL vs MSFT"),
    ("Data diagnostics", "Check ORCA data diagnostics"),
]
chip_cols = st.columns(len(quick))
for col, (label, prompt) in zip(chip_cols, quick):
    with col:
        disabled = should_disable_quick_action(
            label,
            api_offline=api_offline,
            has_symbol=bool(primary_symbol),
            coverage_ready=coverage_ready,
        )
        if st.button(label, use_container_width=True, disabled=disabled, key=f"chip-{label}"):
            S.add_user(prompt)
            with st.spinner("Routing to ORCA..."):
                reply = J.submit(prompt, primary_symbol, horizon, risk, portfolio_metadata=portfolio_metadata)
            if reply:
                S.add_assistant(reply)
            st.rerun()

jobs = S.pending_jobs()
if jobs:
    C.kicker("Active ORCA jobs")
    for job in list(jobs):
        status = S.display_status(job)
        icon = C.STATUS_ICON.get(status, "-")
        jcols = st.columns([0.9, 0.7, 2.2, 1.2, 0.7, 0.7], vertical_alignment="center")
        jcols[0].markdown(f"**{job.get('symbol','?')}**")
        jcols[1].markdown(f"{icon} `{status}`")
        jcols[2].caption(C._truncate(job.get("prompt"), 60))
        jcols[3].caption(f"{S.fmt_elapsed(job.get('created_at'))}")
        if status in {"failed", "stale"} and jcols[4].button("Retry", key=f"retry-{job['job_id']}", help="Retry", disabled=api_offline):
            J.retry_job(job, horizon, risk)
            st.rerun()
        if jcols[5].button("Remove", key=f"rm-{job['job_id']}", help="Remove"):
            S.remove_job(job["job_id"])
            st.rerun()
        if status in {"queued", "running"}:
            with st.spinner(f"Waiting for job {job['job_id'][:8]}..."):
                if job.get("events_complete"):
                    J.poll_job_result(job)
                else:
                    J.stream_events(job)
            st.rerun()

if st.session_state.submit_retry:
    retry_prompt = st.session_state.submit_retry
    c1, c2 = st.columns([1, 0.22])
    c1.warning(f"Last submit failed. Retry: *{C._truncate(retry_prompt, 60)}*")
    if c2.button("Retry", disabled=api_offline):
        S.add_user(retry_prompt)
        reply = J.submit(retry_prompt, primary_symbol, horizon, risk, portfolio_metadata=portfolio_metadata)
        if reply:
            S.add_assistant(reply)
        st.session_state.submit_retry = None
        st.rerun()

S.sync_jobs_to_query()

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

if user_prompt := st.chat_input(
    f"Ask ORCA about {primary_symbol or 'markets'}...",
    disabled=submit_disabled,
):
    S.add_user(user_prompt)
    with st.chat_message("user"):
        st.markdown(user_prompt)

    reply = None
    if api_offline:
        reply = "ORCA API offline. Start the backend first."
    else:
        with st.spinner("Routing to ORCA..."):
            reply = J.submit(user_prompt, primary_symbol, horizon, risk, portfolio_metadata=portfolio_metadata)
        if reply and any(k in reply for k in ("**Api", "**Timeout", "**Malformed")):
            st.session_state.submit_retry = user_prompt

    if reply:
        S.add_assistant(reply)
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        st.rerun()
