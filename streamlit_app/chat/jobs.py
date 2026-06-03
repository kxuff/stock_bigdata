"""Job tracking & submission logic for ORCA AI Chat."""
from __future__ import annotations

import streamlit as st

from services.advisory_api import (
    create_agent_query_job,
    fetch_health,
    fetch_status,
    stream_agent_query_job_events,
    stream_decision_job_events,
)
from chat import state


# ── Horizon / risk maps ───────────────────────────────────────────────────────

def risk_value(label: str) -> str:
    return {"Low": "CONSERVATIVE", "Medium": "MODERATE", "High": "AGGRESSIVE"}.get(label, "MODERATE")


def horizon_value(label: str) -> str:
    return {
        "Intraday": "INTRADAY",
        "1-4 weeks": "SHORT_TERM",
        "1-3 months": "MEDIUM_TERM",
        "6-12 months": "LONG_TERM",
    }.get(label, "SHORT_TERM")


# ── Error formatting ──────────────────────────────────────────────────────────

def _safe_error(exc: Exception) -> str:
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        if isinstance(payload, dict):
            msg = payload.get("message") or payload.get("detail")
            if not msg and isinstance(payload.get("error"), dict):
                msg = payload["error"].get("message")
            if msg:
                return f"ORCA API error ({resp.status_code}): {msg}"
        return f"ORCA API error ({resp.status_code})."
    return f"ORCA API error: {exc.__class__.__name__}"


def _error_md(kind: str, message: str, detail: str | None = None) -> str:
    det = f"\n\n<details><summary>Technical detail</summary>\n\n```text\n{detail}\n```\n</details>" if detail else ""
    return f"**{kind.replace('_',' ').title()}**\n\n{message}{det}"


def _classify(exc: Exception) -> str:
    name = exc.__class__.__name__.lower()
    if "timeout" in name:
        return "timeout"
    return "api_offline" if getattr(exc, "response", None) is None else "api_error"


def _extract_error(payload: dict) -> str:
    body = payload.get("body") if isinstance(payload, dict) else None
    src = body if isinstance(body, dict) else payload
    return str(src.get("message") or src.get("detail") or src.get("error_code") or "ORCA job failed.")


# ── Backend check ─────────────────────────────────────────────────────────────

def check_backend() -> dict:
    try:
        health = fetch_health()
        status = fetch_status()
    except Exception as exc:  # noqa: BLE001
        return {"state": "Offline", "error": _safe_error(exc)}
    health_ok = (health.get("status") or "").lower() in {"ok", "healthy"}
    status_ok = (status.get("status") or "").lower() in {"ok", "healthy", "ready"}
    return {"state": "Connected" if health_ok and status_ok else "Degraded", "error": None}


# ── Submit ────────────────────────────────────────────────────────────────────

def submit(prompt: str, symbol: str, horizon: str, risk: str) -> str | None:
    """Create /api/v1/agent/query-jobs async job, update session state, return error string or None."""
    try:
        fetch_health()
        job = create_agent_query_job({
            "message": prompt,
            "context": {
                "symbol": symbol or None,
                "investment_horizon": horizon_value(horizon),
                "risk_tolerance": risk_value(risk),
            },
        })
        if not isinstance(job, dict) or not job.get("job_id"):
            return _error_md("malformed_response", "ORCA returned no job_id.", repr(job))
        state.add_job({
            "job_id":         job["job_id"],
            "kind":           "agent_query",
            "symbol":         symbol or "N/A",
            "route":          None,
            "prompt":         prompt,
            "created_at":     state.utc_now().isoformat(),
            "updated_at":     None,
            "status":         job.get("status", "queued"),
            "result_fetched": False,
            "events_complete": False,
        })
        return None
    except Exception as exc:  # noqa: BLE001
        return _error_md(_classify(exc), _safe_error(exc), repr(exc))


# ── Stream events ─────────────────────────────────────────────────────────────

def stream_events(job: dict) -> None:
    try:
        if not job.get("job_id"):
            job["status"] = "failed"; job["error_message"] = "Missing job_id."
            return
        event_source = stream_agent_query_job_events if job.get("kind") == "agent_query" else stream_decision_job_events
        for event in event_source(job["job_id"]):
            etype = event.get("event")
            data  = event.get("data") or {}
            if not isinstance(data, dict):
                continue
            if etype == "status":
                job.update({
                    "status":         data.get("status", job.get("status")),
                    "progress_stage": data.get("progress_stage"),
                    "progress_pct":   data.get("progress"),
                    "run_id":         data.get("run_id", job.get("run_id")),
                    "updated_at":     data.get("updated_at") or state.utc_now().isoformat(),
                })
                if state.is_stale(job):
                    job["status"] = "stale"
            elif etype == "result":
                if job.get("kind") == "agent_query":
                    _add_agent_query_result(job["job_id"], data)
                else:
                    state.add_decision_once(f"{job['job_id']}:result", data)
                job.update({"status": "completed", "result_fetched": True,
                            "events_complete": True, "updated_at": state.utc_now().isoformat()})
                break
            elif etype in {"failure", "error"}:
                err_msg = _extract_error(data)
                job.update({"status": "failed", "error": data, "error_message": err_msg,
                            "events_complete": True, "updated_at": state.utc_now().isoformat()})
                state.add_once(f"{job['job_id']}:failed",
                    f"### ORCA job failed: {job.get('symbol','N/A')}\n\n{err_msg}")
                break
        state.sync_jobs_to_query()
    except Exception as exc:  # noqa: BLE001
        job["status"] = "failed"
        job["error_message"] = _safe_error(exc)
        st.error(_safe_error(exc))


def _add_agent_query_result(job_id: str, response: dict) -> None:
    rt = response.get("result_type")
    if rt == "single_symbol_decision":
        state.add_decision_once(f"{job_id}:result", response.get("result") or {})
        return
    if rt:
        state.add_agent_response_once(f"{job_id}:result", response)
        return
    if response.get("suggested_actions"):
        state.add_agent_response_once(f"{job_id}:result", response)
        return
    route = response.get("route", "unknown")
    state.add_once(f"{job_id}:result", f"**Route:** `{route}`\n\n{response.get('message', '')}")


# ── Retry ─────────────────────────────────────────────────────────────────────

def retry_job(job: dict, horizon: str, risk: str) -> None:
    prompt = job.get("prompt")
    symbol = job.get("symbol")
    if not prompt or not symbol:
        st.warning("Retry unavailable — prompt not persisted across reload.")
        return
    state.add_user(prompt)
    reply = submit(prompt, symbol, horizon, risk)
    if reply:
        state.add_assistant(reply)
