"""Job tracking & submission logic for ORCA AI Chat."""
from __future__ import annotations

import streamlit as st

from services.advisory_api import (
    create_agent_query_job,
    fetch_health,
    fetch_status,
    get_agent_query_job_result,
    get_decision_job_result,
    stream_agent_query_job_events,
    stream_decision_job_events,
)
from chat import state


DEFAULT_PORTFOLIO_HOLDINGS = "AAPL:35, MSFT:35, NVDA:20, CASH:10"


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


def parse_holdings_text(value: str) -> tuple[list[dict], list[str]]:
    holdings: list[dict] = []
    warnings: list[str] = []
    for raw_part in (value or "").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            warnings.append(f"Invalid holding '{part}'. Use SYMBOL:weight.")
            continue
        raw_symbol, raw_weight = part.split(":", 1)
        symbol = raw_symbol.strip().upper().replace(".", "-")
        try:
            weight = float(raw_weight.strip())
        except ValueError:
            warnings.append(f"Invalid weight for {symbol or 'holding'}.")
            continue
        if not symbol:
            warnings.append("Holding symbol is blank.")
            continue
        if weight < 0 or weight > 100:
            warnings.append(f"Weight for {symbol} must be between 0 and 100.")
            continue
        holdings.append({"symbol": symbol, "weight": weight})
    return holdings, warnings


def build_portfolio_metadata(
    holdings_text: str,
    *,
    min_cash_weight: float | None = None,
    max_single_asset_weight: float | None = None,
) -> tuple[dict, list[str]]:
    holdings, warnings = parse_holdings_text(holdings_text)
    metadata: dict = {}
    if holdings:
        metadata["holdings"] = holdings
    if min_cash_weight is not None:
        metadata["min_cash_weight"] = float(min_cash_weight)
    if max_single_asset_weight is not None:
        metadata["max_single_asset_weight"] = float(max_single_asset_weight)
    return metadata, warnings


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

def submit(prompt: str, symbol: str, horizon: str, risk: str, portfolio_metadata: dict | None = None) -> str | None:
    """Create /api/v1/agent/query-jobs async job, update session state, return error string or None."""
    try:
        fetch_health()
        context = {
            "symbol": symbol or None,
            "investment_horizon": horizon_value(horizon),
            "risk_tolerance": risk_value(risk),
        }
        if portfolio_metadata:
            context["metadata"] = portfolio_metadata
        job = create_agent_query_job({
            "message": prompt,
            "context": context,
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
            "portfolio_metadata": portfolio_metadata or {},
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
        poll_status = poll_job_result(job)
        if poll_status not in {"completed", "failed"}:
            job["events_complete"] = True
            job["error_message"] = _safe_error(exc)
            st.warning(f"Live job stream unavailable. Polling fallback is active: {_safe_error(exc)}")
        state.sync_jobs_to_query()


def poll_job_result(job: dict) -> str:
    """Poll a job result endpoint once and update local job state."""
    job_id = job.get("job_id")
    if not job_id:
        job["status"] = "failed"
        job["error_message"] = "Missing job_id."
        return "failed"
    try:
        getter = get_agent_query_job_result if job.get("kind") == "agent_query" else get_decision_job_result
        status_code, payload = getter(job_id, timeout=10.0)
    except Exception as exc:  # noqa: BLE001
        job["error_message"] = _safe_error(exc)
        return "error"
    return apply_job_result(job, status_code, payload)


def apply_job_result(job: dict, status_code: int, payload: dict) -> str:
    if status_code == 202:
        job.update(
            {
                "status": payload.get("status", job.get("status", "running")),
                "progress_stage": payload.get("progress_stage"),
                "progress_pct": payload.get("progress"),
                "updated_at": payload.get("updated_at") or state.utc_now().isoformat(),
            }
        )
        if state.is_stale(job):
            job["status"] = "stale"
            state.sync_jobs_to_query()
            return "failed"
        state.sync_jobs_to_query()
        return "pending"
    if status_code == 200:
        if job.get("kind") == "agent_query":
            _add_agent_query_result(job["job_id"], payload)
        else:
            state.add_decision_once(f"{job['job_id']}:result", payload)
        job.update(
            {
                "status": "completed",
                "result_fetched": True,
                "events_complete": True,
                "updated_at": state.utc_now().isoformat(),
            }
        )
        state.sync_jobs_to_query()
        return "completed"
    job.update(
        {
            "status": "failed",
            "error": payload,
            "error_message": _extract_error(payload),
            "events_complete": True,
            "updated_at": state.utc_now().isoformat(),
        }
    )
    state.sync_jobs_to_query()
    return "failed"


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
        st.warning("Retry unavailable - prompt not persisted across reload.")
        return
    state.add_user(prompt)
    reply = submit(prompt, symbol, horizon, risk, portfolio_metadata=job.get("portfolio_metadata") or None)
    if reply:
        state.add_assistant(reply)
