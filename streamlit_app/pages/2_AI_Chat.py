from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from time import sleep
from uuid import uuid4

import streamlit as st

from services.advisory_api import create_decision_job, fetch_health, fetch_readiness, fetch_status, get_decision_job, get_decision_job_result


st.set_page_config(page_title="AI Chat", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --orca-bg: #020617;
        --orca-panel: rgba(15, 23, 42, 0.82);
        --orca-panel-strong: rgba(8, 13, 28, 0.96);
        --orca-border: rgba(103, 232, 249, 0.18);
        --orca-cyan: #67e8f9;
        --orca-emerald: #6ee7b7;
        --orca-amber: #fcd34d;
        --orca-muted: #94a3b8;
        --orca-text: #e5e7eb;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(20, 184, 166, 0.20), transparent 34rem),
            radial-gradient(circle at 86% 18%, rgba(34, 211, 238, 0.16), transparent 32rem),
            linear-gradient(135deg, #020617 0%, #07111f 48%, #0f172a 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(8, 47, 73, 0.62));
        border-right: 1px solid rgba(103, 232, 249, 0.18);
    }

    .orca-hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.2rem;
        margin-bottom: 1.2rem;
        border: 1px solid var(--orca-border);
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(6, 78, 59, 0.92), rgba(8, 13, 28, 0.88) 52%, rgba(22, 78, 99, 0.72)),
            repeating-linear-gradient(90deg, rgba(255,255,255,0.035) 0 1px, transparent 1px 16px);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
    }

    .orca-hero:after {
        content: "";
        position: absolute;
        inset: auto -8% -40% 45%;
        height: 210px;
        background: radial-gradient(circle, rgba(103, 232, 249, 0.34), transparent 68%);
        filter: blur(8px);
    }

    .orca-eyebrow {
        color: var(--orca-cyan);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    .orca-title {
        margin: 0.25rem 0 0.35rem;
        color: #f8fafc;
        font-size: clamp(2.2rem, 5vw, 4.2rem);
        font-weight: 900;
        line-height: 0.94;
        letter-spacing: -0.075em;
    }

    .orca-subtitle {
        max-width: 760px;
        color: #cbd5e1;
        font-size: 1.02rem;
        line-height: 1.65;
    }

    .orca-badge-row { margin-top: 1.1rem; display: flex; flex-wrap: wrap; gap: 0.55rem; }
    .orca-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.38rem;
        padding: 0.42rem 0.72rem;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(2, 6, 23, 0.42);
        color: #e2e8f0;
        font-size: 0.82rem;
        font-weight: 800;
    }

    .market-card, .prompt-card {
        min-height: 126px;
        padding: 1rem 1.05rem;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 20px;
        background: linear-gradient(160deg, rgba(15, 23, 42, 0.88), rgba(8, 47, 73, 0.36));
        box-shadow: 0 18px 48px rgba(2, 6, 23, 0.26);
    }

    .market-label {
        color: var(--orca-muted);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .market-value {
        margin-top: 0.35rem;
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 900;
    }

    .market-note { margin-top: 0.45rem; color: #a7f3d0; font-size: 0.88rem; }

    .section-kicker {
        margin: 1.25rem 0 0.55rem;
        color: var(--orca-cyan);
        font-weight: 900;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.76rem;
    }

    .prompt-card {
        min-height: 88px;
        border-color: rgba(103, 232, 249, 0.22);
        background: linear-gradient(150deg, rgba(8, 47, 73, 0.64), rgba(15, 23, 42, 0.86));
    }

    .prompt-card b { color: #f8fafc; }
    .prompt-card span { color: var(--orca-muted); font-size: 0.88rem; }

    .stButton > button {
        border: 1px solid rgba(103, 232, 249, 0.34);
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(20, 184, 166, 0.24), rgba(34, 211, 238, 0.12));
        color: #e0f2fe;
        font-weight: 800;
        transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(103, 232, 249, 0.72);
        background: linear-gradient(90deg, rgba(20, 184, 166, 0.36), rgba(34, 211, 238, 0.20));
    }

    [data-testid="stChatMessage"] {
        border-radius: 22px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: rgba(15, 23, 42, 0.64);
        box-shadow: 0 14px 36px rgba(2, 6, 23, 0.18);
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 18px;
        border-color: rgba(103, 232, 249, 0.34);
        background: rgba(2, 6, 23, 0.72);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _risk_value(value: str) -> str:
    return {"Low": "CONSERVATIVE", "Medium": "MODERATE", "High": "AGGRESSIVE"}[value]


def _horizon_value(value: str) -> str:
    return {"Intraday": "INTRADAY", "1-4 weeks": "SHORT_TERM", "1-3 months": "MEDIUM_TERM", "6-12 months": "LONG_TERM"}[value]


def _request_payload(prompt: str, symbol: str, horizon_value: str, risk_value: str) -> dict:
    now = datetime.now(UTC).isoformat()
    return {
        "request_id": f"streamlit-{uuid4()}",
        "timestamp": now,
        "as_of_timestamp": now,
        "user_query": prompt,
        "decision_mode": "single_symbol_advisory",
        "symbols": [symbol],
        "user_context": {"risk_tolerance": risk_value, "investment_horizon": horizon_value},
        "metadata": {"source": "streamlit_ai_chat"},
    }


def _format_decision(decision: dict) -> str:
    rationale = decision.get("decision_rationale") or []
    bullets = "\n".join(f"- {item.get('factor', 'Factor')}: {item.get('explanation', '')}" for item in rationale[:4])
    warnings = "\n".join(f"- {item}" for item in (decision.get("risk_warnings") or [])[:4]) or "- None reported"
    return f"""
### ORCA Decision: {decision.get('symbol', 'N/A')}
**Recommendation:** {decision.get('recommendation', 'N/A')}  
**Confidence:** {decision.get('confidence', 'N/A')}  
**Human review:** {decision.get('requires_human_review', False)}  
**Run ID:** `{decision.get('run_id', 'N/A')}`

{decision.get('summary', '')}

**Rationale**
{bullets or '- No rationale returned'}

**Risk warnings**
{warnings}

_Not financial advice._
"""


def _readiness_failure_summary(readiness: dict) -> dict:
    failed = {}
    for name, tool in (readiness.get("tools") or {}).items():
        status_value = tool.get("status")
        missing_symbols = tool.get("missing_symbols") or []
        is_stale = (tool.get("freshness") or {}).get("is_stale")
        if status_value != "SUCCESS" or missing_symbols or is_stale:
            failed[name] = {
                "status": status_value,
                "missing_symbols": missing_symbols,
                "is_stale": is_stale,
            }
    if readiness.get("error"):
        failed["provider"] = {"status": "ERROR"}
    return failed


def _safe_api_error(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if isinstance(payload, dict):
            message = payload.get("message") or payload.get("detail")
            if not message and isinstance(payload.get("error"), dict):
                message = payload["error"].get("message")
            if message:
                return f"ORCA API error ({response.status_code}): {message}"
        return f"ORCA API error ({response.status_code})."
    return f"ORCA API error: {exc.__class__.__name__}"


def _run_orca_flow(prompt: str, symbol: str, horizon_value: str, risk_value: str) -> str:
    if not symbol:
        return "ORCA API error: select at least one symbol."
    try:
        fetch_health()
        fetch_status()
        readiness = fetch_readiness([symbol])
        if not readiness.get("ready"):
            return f"ORCA data not ready for {symbol}. Blocking checks: `{_readiness_failure_summary(readiness)}`"
        job = create_decision_job(_request_payload(prompt, symbol, _horizon_value(horizon_value), _risk_value(risk_value)))
        job_id = job["job_id"]
        for _ in range(24):
            status_payload = get_decision_job(job_id)
            if status_payload.get("status") in {"succeeded", "failed"}:
                break
            sleep(1)
        result_status, result = get_decision_job_result(job_id)
        if result_status == 202:
            return f"ORCA job `{job_id}` still running. Try again soon."
        return _format_decision(result)
    except Exception as exc:  # noqa: BLE001 - UI must show API failure, no fake content.
        return _safe_api_error(exc)

st.markdown(
    """
    <div class="orca-hero">
      <div class="orca-eyebrow">ORCA market intelligence · production cockpit</div>
      <div class="orca-title">Ask sharper.<br/>Trade calmer.</div>
      <div class="orca-subtitle">
        Scenario chat for symbols, horizons, and risk posture. Answers route through production ORCA API.
      </div>
      <div class="orca-badge-row">
        <span class="orca-badge">● Production API</span>
        <span class="orca-badge">CrewAI advisory jobs</span>
        <span class="orca-badge">Market copilot</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("ORCA Context")
    st.caption("Frame advisory requests for backend services.")
    symbols = st.text_input("Symbols", "NVDA, MSFT, LLY")
    horizon = st.selectbox("Horizon", ["Intraday", "1-4 weeks", "1-3 months", "6-12 months"], index=1)
    risk = st.select_slider("Risk tolerance", ["Low", "Medium", "High"], value="Medium")
    st.markdown("---")
    st.markdown("<span style='color:#67e8f9;font-weight:800'>● Production mode</span>", unsafe_allow_html=True)
    st.caption("Prompts create bounded async advisory jobs.")

if "messages" not in st.session_state:
    st.session_state.messages = []

symbol_list = [symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()]
symbol_display = ", ".join(symbol_list) or "No symbols"
primary_symbol = symbol_list[0] if symbol_list else ""
if len(symbol_list) > 1:
    primary_symbol = st.selectbox("Primary symbol for advisory", symbol_list)

summary_cols = st.columns([1.25, 1, 1])
summary_cards = [
    ("Active symbols", escape(symbol_display), "Universe checked for data readiness"),
    ("Time horizon", escape(horizon), "Answer lens for setup and catalysts"),
    ("Risk posture", escape(risk), "Controls sizing tone and caution flags"),
]
for col, (label, value, note) in zip(summary_cols, summary_cards):
    col.markdown(
        f"""
        <div class="market-card">
          <div class="market-label">{label}</div>
          <div class="market-value">{value}</div>
          <div class="market-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-kicker">Launch questions</div>', unsafe_allow_html=True)

prompt_cols = st.columns(3)
sample_prompts = [
    ("Single-symbol setup", "Analyze the selected symbol setup", "Trend, catalysts, risk."),
    ("Risk audit", "Show risks for the selected symbol", "Position sizing and watch items."),
    ("Decision rationale", "Explain the selected symbol recommendation", "Signals, conflicts, confidence."),
]
for col, (label, sample, detail) in zip(prompt_cols, sample_prompts):
    col.markdown(
        f"""
        <div class="prompt-card">
          <b>{label}</b><br/>
          <span>{detail}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if col.button(sample, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": sample})
        with st.spinner("ORCA advisory job running..."):
            st.session_state.messages.append({"role": "assistant", "content": _run_orca_flow(sample, primary_symbol, horizon, risk)})

st.markdown('<div class="section-kicker">Conversation tape</div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask ORCA about markets or stocks..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    with st.spinner("ORCA advisory job running..."):
        reply = _run_orca_flow(user_prompt, primary_symbol, horizon, risk)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
