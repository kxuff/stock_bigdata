from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import streamlit as st

from services.advisory_api import create_agent_query, fetch_health, fetch_status, stream_decision_job_events


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

    .section-kicker {
        margin: 0.8rem 0 0.35rem;
        color: var(--orca-cyan);
        font-weight: 900;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.76rem;
    }

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


def _normalize_symbol(raw_symbol: str) -> str:
    return raw_symbol.strip().upper().replace(".", "-")


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


def _decision_summary(decision: dict) -> str:
    return (
        f"{decision.get('symbol', 'N/A')} · {decision.get('recommendation', 'N/A')} · "
        f"confidence {decision.get('confidence', 'N/A')}\n\n{decision.get('summary', '')}"
    )


def _render_decision(decision: dict) -> None:
    symbol = decision.get("symbol", "N/A")
    recommendation = decision.get("recommendation", "N/A")
    confidence = decision.get("confidence", "N/A")
    header_cols = st.columns(3)
    header_cols[0].metric("Symbol", symbol)
    header_cols[1].metric("Recommendation", recommendation)
    header_cols[2].metric("Confidence", confidence)
    if decision.get("requires_human_review"):
        st.warning("Human review required before action.")
    summary = decision.get("summary") or "No summary returned."
    st.info(summary)

    rationale_rows = []
    for item in decision.get("decision_rationale") or []:
        if isinstance(item, dict):
            rationale_rows.append(
                {
                    "Factor": item.get("factor", "Factor"),
                    "Stance": item.get("stance", "—"),
                    "Weight": item.get("weight", "—"),
                    "Explanation": item.get("explanation", ""),
                }
            )
    if rationale_rows:
        st.table(rationale_rows)

    warnings = decision.get("risk_warnings") or []
    if warnings:
        st.error("\n".join(f"• {item}" for item in warnings))
    else:
        st.success("No risk warnings returned.")

    support_col, conflict_col = st.columns(2)
    supporting = decision.get("supporting_signals") or decision.get("supporting_evidence") or []
    conflicting = decision.get("conflicting_signals") or decision.get("conflicts") or []
    support_col.write("**Supporting signals**")
    support_col.write(supporting or "—")
    conflict_col.write("**Conflicting signals**")
    conflict_col.write(conflicting or "—")

    with st.expander("Citations / audit"):
        st.json(
            {
                "run_id": decision.get("run_id"),
                "citations": decision.get("citations") or decision.get("sources") or [],
                "audit": decision.get("audit") or decision.get("audit_trail") or {},
            }
        )

    st.text_area("Copy summary", _decision_summary(decision), height=110, key=f"summary-{decision.get('run_id', uuid4())}")


def _render_agent_response(response: dict) -> None:
    result_type = response.get("result_type")
    result = response.get("result") or {}
    st.caption(f"Route: {response.get('route', 'unknown')} · confidence {response.get('router_confidence', 0):.2f}")
    if response.get("message"):
        st.info(response["message"])
    if result_type == "single_symbol_decision":
        _render_decision(result)
        return
    if result_type in {
        "symbol_comparison",
        "universe_screen",
        "watchlist_review",
        "market_brief",
        "data_diagnostics",
        "portfolio_rebalance",
        "backtest_analysis",
        "streaming_pipeline_health",
        "streaming_freshness_check",
        "streaming_alert_review",
        "streaming_symbol_monitor",
        "streaming_feature_drift",
        "streaming_ingestion_lag",
        "streaming_topic_inspection",
        "streaming_quality_incidents",
    }:
        _render_structured_agent_result(result_type, result)
    actions = response.get("suggested_actions") or []
    if actions:
        st.markdown("**Suggested actions**")
        for action in actions[:5]:
            st.markdown(f"- {action.get('label', action)}")


def _render_structured_agent_result(result_type: str, result: dict) -> None:
    if result_type == "symbol_comparison":
        rows = result.get("rows") or []
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        return
    if result_type == "universe_screen":
        rows = result.get("candidates") or []
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        diagnostics = result.get("diagnostics") or {}
        if diagnostics:
            with st.expander("Diagnostics"):
                st.json(diagnostics)
        return
    if result_type == "watchlist_review":
        rows = result.get("items") or []
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        return
    if result_type == "market_brief":
        if result.get("summary"):
            st.write(result["summary"])
        leaders = result.get("leaders") or []
        if leaders:
            st.dataframe(leaders, width="stretch", hide_index=True)
        return
    if result_type == "data_diagnostics":
        st.json(result.get("diagnostics") or result)
        return
    if result_type == "portfolio_rebalance":
        if result.get("message"):
            st.warning(result["message"])
        changes = result.get("changes") or []
        if changes:
            st.dataframe(changes, width="stretch", hide_index=True)
        cols = st.columns(2)
        cols[0].metric("Cash target", result.get("cash_target_weight", 0))
        cols[1].metric("Human review", str(result.get("human_review_required", True)))
        with st.expander("Constraints"):
            st.json(result.get("constraints") or {})
        return
    if result_type == "backtest_analysis":
        st.info(result.get("limitation") or "Backtest service not connected.")
        if result.get("suggested_next_action"):
            st.success(result["suggested_next_action"])
        with st.expander("Backtest spec", expanded=True):
            st.json(result.get("backtest_spec") or {})
        return
    streaming_tables = {
        "streaming_pipeline_health": "stages",
        "streaming_freshness_check": "rows",
        "streaming_alert_review": "alerts",
        "streaming_feature_drift": "rows",
        "streaming_ingestion_lag": "rows",
        "streaming_topic_inspection": "samples",
        "streaming_quality_incidents": "incidents",
    }
    if result_type == "streaming_symbol_monitor":
        if result.get("symbol"):
            st.metric("Symbol", result["symbol"])
        freshness = result.get("freshness") or []
        alerts = result.get("alerts") or []
        if freshness:
            st.markdown("**Freshness**")
            st.dataframe(freshness, width="stretch", hide_index=True)
        if alerts:
            st.markdown("**Active alerts**")
            st.dataframe(alerts, width="stretch", hide_index=True)
        if not freshness and not alerts:
            st.json(result)
        return
    if result_type in streaming_tables:
        rows = result.get(streaming_tables[result_type]) or []
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.json(result)
        return


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


def _readiness_failure_rows(readiness: dict) -> list[dict]:
    rows = []
    for name, tool in (readiness.get("tools") or {}).items():
        status_value = tool.get("status")
        missing_symbols = tool.get("missing_symbols") or []
        is_stale = (tool.get("freshness") or {}).get("is_stale")
        if status_value != "SUCCESS" or missing_symbols or is_stale:
            rows.append(
                {
                    "Tool": name,
                    "Status": status_value or "UNKNOWN",
                    "Missing symbols": ", ".join(missing_symbols) or "—",
                    "Stale": "Yes" if is_stale else "No",
                }
            )
    if readiness.get("error"):
        rows.append({"Tool": "provider", "Status": "ERROR", "Missing symbols": "—", "Stale": "—"})
    return rows


def _format_readiness_failure(symbol: str, readiness: dict) -> str:
    rows = _readiness_failure_rows(readiness)
    if not rows:
        return f"ORCA data not ready for `{symbol}`."
    table = "| Tool | Status | Missing symbols | Stale |\n|---|---|---|---|\n"
    table += "\n".join(f"| {row['Tool']} | {row['Status']} | {row['Missing symbols']} | {row['Stale']} |" for row in rows)
    return f"ORCA data not ready for `{symbol}`.\n\n{table}"


def _check_backend() -> dict:
    try:
        health = fetch_health()
        status = fetch_status()
    except Exception as exc:  # noqa: BLE001 - page must stay usable while API offline.
        return {"state": "Offline", "health": None, "status": None, "error": _safe_api_error(exc)}
    health_ok = (health.get("status") or "").lower() in {"ok", "healthy"}
    status_ok = (status.get("status") or "").lower() in {"ok", "healthy", "ready"}
    state = "Connected" if health_ok and status_ok else "Degraded"
    return {"state": state, "health": health, "status": status, "error": None}


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


def _classify_exception(exc: Exception) -> str:
    name = exc.__class__.__name__.lower()
    if "timeout" in name:
        return "timeout"
    response = getattr(exc, "response", None)
    if response is None:
        return "api_offline"
    return "api_error"


def _extract_error_message(payload: dict) -> str:
    body = payload.get("body") if isinstance(payload, dict) else None
    if isinstance(body, dict):
        return str(body.get("message") or body.get("detail") or body.get("error_code") or "ORCA job failed.")
    return str(payload.get("message") or payload.get("detail") or payload.get("error_code") or "ORCA job failed.")


def _error_markdown(kind: str, message: str, detail: str | None = None) -> str:
    detail_block = f"\n\n<details><summary>Technical detail</summary>\n\n```text\n{detail}\n```\n</details>" if detail else ""
    return f"**{kind.replace('_', ' ').title()}**\n\n{message}{detail_block}"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_time(value: str | None) -> str:
    parsed = _parse_iso_datetime(value)
    if not parsed:
        return "—"
    return parsed.strftime("%H:%M:%S UTC")


def _format_elapsed(start_value: str | None, end: datetime | None = None) -> str:
    start = _parse_iso_datetime(start_value)
    if not start:
        return "—"
    seconds = max(0, int(((end or _utc_now()) - start).total_seconds()))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _is_stale_job(job: dict) -> bool:
    if job.get("status") == "stale":
        return True
    created_at = _parse_iso_datetime(job.get("created_at"))
    if not created_at:
        return False
    return job.get("status") in {"queued", "running"} and _utc_now() - created_at > timedelta(hours=1)


def _display_status(job: dict) -> str:
    status = job.get("status", "unknown")
    if status in {"succeeded", "success"}:
        status = "completed"
    return "stale" if _is_stale_job(job) else status


def _format_progress(job: dict) -> str:
    stage = job.get("progress_stage") or job.get("status") or "Progress"
    pct = job.get("progress_pct")
    if pct is None:
        return str(stage)
    return f"{stage} · {pct}%"


def _status_icon(status: str) -> str:
    return {"queued": "🕓", "running": "🔄", "completed": "✅", "failed": "❌", "stale": "⚠️"}.get(status, "•")


def _truncate(value: str | None, limit: int = 64) -> str:
    text = value or "—"
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _pending_jobs() -> list[dict]:
    if "pending_orca_jobs" not in st.session_state:
        st.session_state.pending_orca_jobs = _load_pending_jobs_from_query()
    return st.session_state.pending_orca_jobs


def _load_pending_jobs_from_query() -> list[dict]:
    encoded = st.query_params.get("orca_jobs")
    if not encoded:
        return []
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        jobs = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
    except (ValueError, TypeError, json.JSONDecodeError):
        return []
    if not isinstance(jobs, list):
        return []
    safe_jobs = []
    for job in jobs:
        if isinstance(job, dict) and job.get("job_id"):
            safe_jobs.append(
                {
                    "job_id": job.get("job_id"),
                    "symbol": job.get("symbol"),
                    "created_at": job.get("created_at"),
                    "status": job.get("status", "queued"),
                }
            )
    return safe_jobs


def _sync_pending_jobs_to_query() -> None:
    jobs = _pending_jobs()
    if not jobs:
        if "orca_jobs" in st.query_params:
            del st.query_params["orca_jobs"]
        return
    compact_jobs = [
        {
            "job_id": job.get("job_id"),
            "symbol": job.get("symbol"),
            "created_at": job.get("created_at"),
            "status": job.get("status"),
        }
        for job in jobs
    ]
    encoded = base64.urlsafe_b64encode(json.dumps(compact_jobs, separators=(",", ":")).encode()).decode().rstrip("=")
    st.query_params["orca_jobs"] = encoded


def _append_assistant_message_once(message_id: str, content: str) -> None:
    if message_id in st.session_state.completed_orca_message_ids:
        return
    st.session_state.messages.append({"role": "assistant", "content": content})
    st.session_state.completed_orca_message_ids.add(message_id)


def _append_decision_once(message_id: str, decision: dict) -> None:
    if message_id in st.session_state.completed_orca_message_ids:
        return
    st.session_state.messages.append({"role": "assistant", "type": "decision", "decision": decision})
    st.session_state.completed_orca_message_ids.add(message_id)


def _remove_pending_job(job_id: str) -> None:
    st.session_state.pending_orca_jobs = [job for job in _pending_jobs() if job.get("job_id") != job_id]
    _sync_pending_jobs_to_query()


def _job_error_message(job: dict, status_payload: dict | None = None) -> str:
    payload = status_payload or job
    error = payload.get("error") or {}
    error_code = payload.get("error_code") or error.get("code") or "unknown_error"
    error_message = payload.get("error_message") or error.get("message") or "ORCA job failed."
    return f"""
### ORCA job failed: {job.get('symbol', 'N/A')}
**Job ID:** `{job.get('job_id', 'N/A')}`  
**Status:** `{payload.get('status', 'failed')}`  
**Error code:** `{error_code}`

{error_message}
"""


def _submit_orca_job(prompt: str, symbol: str, horizon_value: str, risk_value: str) -> str | None:
    try:
        fetch_health()
        fetch_status()
        response = create_agent_query(
            {
                "message": prompt,
                "context": {
                    "symbol": symbol or None,
                    "investment_horizon": _horizon_value(horizon_value),
                    "risk_tolerance": _risk_value(risk_value),
                },
            }
        )
        if response.get("status") == "immediate":
            if response.get("result_type") == "single_symbol_decision":
                st.session_state.messages.append({"role": "assistant", "type": "decision", "decision": response.get("result") or {}})
                return None
            if response.get("result_type"):
                st.session_state.messages.append({"role": "assistant", "type": "agent_response", "response": response})
                return None
            suggestions = response.get("suggested_actions") or []
            suggestion_lines = "\n".join(f"- {item.get('label', item)}" for item in suggestions[:4])
            route = response.get("route", "unknown")
            suffix = f"\n\n**Suggested actions**\n{suggestion_lines}" if suggestion_lines else ""
            return f"**Route:** `{route}`\n\n{response.get('message', '')}{suffix}"
        job = response.get("job")
        if not isinstance(job, dict) or not job.get("job_id"):
            return _error_markdown("malformed_response", "ORCA returned job response without job_id.", repr(job))
        job_id = job["job_id"]
        job_symbol = (response.get("symbols") or [symbol or "N/A"])[0]
        _pending_jobs().append(
            {
                "job_id": job_id,
                "symbol": job_symbol,
                "route": response.get("route"),
                "prompt": prompt,
                "created_at": _utc_now().isoformat(),
                "updated_at": None,
                "status": job.get("status", "queued"),
                "result_fetched": False,
                "events_complete": False,
            }
        )
        _sync_pending_jobs_to_query()
        return None
    except Exception as exc:  # noqa: BLE001 - UI must show API failure, no fake content.
        kind = _classify_exception(exc)
        return _error_markdown(kind, _safe_api_error(exc), repr(exc))


def _apply_status_event(job: dict, payload: dict) -> None:
    job.update(
        {
            "status": payload.get("status", job.get("status", "unknown")),
            "progress_stage": payload.get("progress_stage"),
            "progress_pct": payload.get("progress"),
            "run_id": payload.get("run_id", job.get("run_id")),
            "updated_at": payload.get("updated_at") or _utc_now().isoformat(),
        }
    )
    if _display_status(job) == "stale":
        job["status"] = "stale"


def _stream_job_events(job: dict) -> None:
    try:
        if not job.get("job_id"):
            job["status"] = "failed"
            job["error_message"] = "Missing job_id."
            return
        for event in stream_decision_job_events(job["job_id"]):
            event_type = event.get("event")
            data = event.get("data") or {}
            if not isinstance(data, dict):
                continue
            if event_type == "status":
                _apply_status_event(job, data)
            elif event_type == "result":
                _append_decision_once(f"{job['job_id']}:result", data)
                job["status"] = "completed"
                job["result_fetched"] = True
                job["events_complete"] = True
                job["updated_at"] = _utc_now().isoformat()
                break
            elif event_type in {"failure", "error"}:
                job["status"] = "failed"
                job["error"] = data
                job["error_message"] = _extract_error_message(data)
                job["events_complete"] = True
                job["updated_at"] = _utc_now().isoformat()
                _append_assistant_message_once(f"{job['job_id']}:failed", _job_error_message(job, {"error": data, "status": "failed"}))
                break
        _sync_pending_jobs_to_query()
    except Exception as exc:  # noqa: BLE001 - status refresh should not crash UI.
        job["status"] = "failed"
        job["error_message"] = _safe_api_error(exc)
        st.error(_safe_api_error(exc))


def _retry_job(job: dict, horizon_value: str, risk_value: str) -> None:
    prompt = job.get("prompt")
    symbol = job.get("symbol")
    if not prompt or not symbol:
        st.warning("Retry unavailable after reload because prompt is kept only in session.")
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    reply = _submit_orca_job(prompt, symbol, horizon_value, risk_value)
    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply})


def _render_pending_jobs() -> None:
    jobs = _pending_jobs()
    if not jobs:
        return

    st.markdown('<div class="section-kicker">ORCA jobs</div>', unsafe_allow_html=True)

    for job in list(jobs):
        status = _display_status(job)
        cols = st.columns([1.15, 0.8, 1.1, 0.9, 0.9, 1.0, 0.8, 0.8], vertical_alignment="center")
        cols[0].markdown(f"**{job.get('symbol', 'N/A')}** `{job.get('job_id', 'N/A')}`")
        cols[1].markdown(f"{_status_icon(status)} `{status}`")
        cols[2].caption(_truncate(job.get("prompt"), 56))
        cols[3].caption(f"Created {_format_time(job.get('created_at'))}")
        cols[4].caption(f"Refreshed {_format_time(job.get('updated_at'))}")
        cols[5].caption(f"Elapsed {_format_elapsed(job.get('created_at'))} · {_format_progress(job)}")
        if status in {"failed", "stale", "completed"}:
            if status in {"failed", "stale"} and cols[6].button("Retry", key=f"retry-{job['job_id']}", use_container_width=True):
                _retry_job(job, horizon, risk)
                st.rerun()
            if cols[7].button("Remove", key=f"remove-{job['job_id']}", use_container_width=True):
                _remove_pending_job(job["job_id"])
                st.rerun()

        if status in {"queued", "running"} and not job.get("events_complete"):
            with st.spinner(f"Listening for ORCA job {job.get('job_id')} events..."):
                _stream_job_events(job)
            st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "completed_orca_message_ids" not in st.session_state:
    st.session_state.completed_orca_message_ids = set()
if "submit_retry" not in st.session_state:
    st.session_state.submit_retry = None
if "orca_backend_status" not in st.session_state:
    st.session_state.orca_backend_status = _check_backend()
_pending_jobs()
_sync_pending_jobs_to_query()

st.title("AI Chat")
st.caption("Scenario chat for symbols, horizons, and risk posture. Answers route through production ORCA API.")

with st.sidebar:
    st.header("ORCA Context")
    st.caption("Frame advisory requests for backend services.")
    symbol_input = st.text_input("Default symbol context", "NVDA")
    horizon = st.selectbox("Horizon", ["Intraday", "1-4 weeks", "1-3 months", "6-12 months"], index=1)
    risk = st.select_slider("Risk tolerance", ["Low", "Medium", "High"], value="Medium")
    st.markdown("---")
    if st.button("Refresh backend", use_container_width=True):
        st.session_state.orca_backend_status = _check_backend()
        st.rerun()
    backend_state = st.session_state.get("orca_backend_status", {}).get("state", "Offline")
    if backend_state == "Connected":
        st.success("Connected")
    elif backend_state == "Degraded":
        st.warning("Degraded")
    else:
        st.error("Offline")
    backend_error = st.session_state.get("orca_backend_status", {}).get("error")
    if backend_error:
        st.caption(backend_error)
    st.caption("Prompts create bounded async advisory jobs.")

backend_status = st.session_state.orca_backend_status
api_offline = backend_status.get("state") == "Offline"

primary_symbol = _normalize_symbol(symbol_input)
if api_offline:
    st.error("ORCA API offline. Start backend before submitting advisory jobs.")

st.markdown('<div class="section-kicker">Launch questions</div>', unsafe_allow_html=True)

prompt_cols = st.columns(3)
sample_prompts = [
    ("Analyze setup", "Analyze the selected symbol setup"),
    ("Show risks", "Show risks for the selected symbol"),
    ("Explain recommendation", "Explain the selected symbol recommendation"),
]
for col, (label, sample) in zip(prompt_cols, sample_prompts):
    if col.button(label, use_container_width=True, disabled=api_offline, help=sample):
        st.session_state.messages.append({"role": "user", "content": sample})
        with st.spinner("Submitting ORCA advisory job..."):
            reply = _submit_orca_job(sample, primary_symbol, horizon, risk)
            if reply:
                st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

if st.session_state.submit_retry:
    retry_prompt = st.session_state.submit_retry
    if st.button("Retry last submit", use_container_width=False, disabled=api_offline):
        st.session_state.messages.append({"role": "user", "content": retry_prompt})
        with st.spinner("Submitting ORCA advisory job..."):
            reply = _submit_orca_job(retry_prompt, primary_symbol, horizon, risk)
            if reply:
                st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.submit_retry = None
        st.rerun()

_render_pending_jobs()

conversation_col, clear_col = st.columns([1, 0.18], vertical_alignment="center")
conversation_col.markdown('<div class="section-kicker">Conversation tape</div>', unsafe_allow_html=True)
if clear_col.button("Clear chat", use_container_width=True):
    st.session_state.messages = []
    st.session_state.completed_orca_message_ids = set()
    st.rerun()

if not st.session_state.messages:
    st.info("No conversation yet. Choose a launch question or ask ORCA below.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "decision":
                _render_decision(message.get("decision") or {})
            elif message.get("type") == "agent_response":
                _render_agent_response(message.get("response") or {})
            else:
                st.markdown(message.get("content", ""))

if user_prompt := st.chat_input("Ask ORCA about markets or stocks..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    reply = None
    if api_offline:
        reply = "ORCA API offline. Start backend before submitting advisory jobs."
    else:
        with st.spinner("Submitting ORCA advisory job..."):
            reply = _submit_orca_job(user_prompt, primary_symbol, horizon, risk)
            if reply and (
                "**Api Offline**" in reply
                or "**Timeout**" in reply
                or "**Malformed Response**" in reply
                or "**Api Error**" in reply
            ):
                st.session_state.submit_retry = user_prompt
    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        st.rerun()
