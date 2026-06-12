"""Session state helpers for ORCA AI Chat."""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
from uuid import uuid4
from datetime import UTC, datetime, timedelta

import streamlit as st


CHAT_STORE_DIR = Path(__file__).resolve().parents[2] / ".streamlit_chat_state"
CHAT_REDIS_URL = os.getenv("STREAMLIT_REDIS_URL") or os.getenv("ORCA_REDIS_URL") or "redis://localhost:6379/0"
CHAT_TTL_SECONDS = int(os.getenv("STREAMLIT_CHAT_TTL_SECONDS", str(60 * 60 * 24 * 7)))


# ── Init ─────────────────────────────────────────────────────────────────────

def init() -> None:
    chat_id = _ensure_chat_id()
    persisted = _load_chat(chat_id)
    defaults: dict = {
        "messages": persisted.get("messages", []),
        "completed_orca_message_ids": set(persisted.get("completed_orca_message_ids", [])),
        "submit_retry": None,
        "orca_backend_status": None,
        "pending_orca_jobs": None,
        "orca_chat_id": chat_id,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Messages ──────────────────────────────────────────────────────────────────

def add_user(content: str) -> None:
    st.session_state.messages.append({"role": "user", "content": content})
    save_chat()


def add_assistant(content: str) -> None:
    st.session_state.messages.append({"role": "assistant", "content": content})
    save_chat()


def add_decision(decision: dict) -> None:
    st.session_state.messages.append({"role": "assistant", "type": "decision", "decision": decision})
    save_chat()


def add_agent_response(response: dict) -> None:
    st.session_state.messages.append({"role": "assistant", "type": "agent_response", "response": response})
    save_chat()


def add_agent_response_once(message_id: str, response: dict) -> None:
    if message_id in st.session_state.completed_orca_message_ids:
        return
    add_agent_response(response)
    st.session_state.completed_orca_message_ids.add(message_id)
    save_chat()


def add_once(message_id: str, content: str) -> None:
    if message_id in st.session_state.completed_orca_message_ids:
        return
    add_assistant(content)
    st.session_state.completed_orca_message_ids.add(message_id)
    save_chat()


def add_decision_once(message_id: str, decision: dict) -> None:
    if message_id in st.session_state.completed_orca_message_ids:
        return
    add_decision(decision)
    st.session_state.completed_orca_message_ids.add(message_id)
    save_chat()


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.completed_orca_message_ids = set()
    save_chat()


def save_chat() -> None:
    chat_id = st.session_state.get("orca_chat_id") or _ensure_chat_id()
    payload = {
        "messages": st.session_state.get("messages", []),
        "completed_orca_message_ids": sorted(st.session_state.get("completed_orca_message_ids", set())),
        "updated_at": utc_now().isoformat(),
    }
    raw = json.dumps(payload, ensure_ascii=False)
    client = _redis_client()
    if client is not None:
        try:
            client.setex(_chat_key(chat_id), CHAT_TTL_SECONDS, raw)
            return
        except Exception:  # noqa: BLE001
            pass
    CHAT_STORE_DIR.mkdir(parents=True, exist_ok=True)
    _chat_path(chat_id).write_text(raw, encoding="utf-8")


def _ensure_chat_id() -> str:
    chat_id = st.query_params.get("orca_chat")
    if not chat_id:
        chat_id = uuid4().hex
        st.query_params["orca_chat"] = chat_id
    return chat_id


def _chat_path(chat_id: str) -> Path:
    safe_id = "".join(ch for ch in chat_id if ch.isalnum() or ch in {"-", "_"})[:64]
    return CHAT_STORE_DIR / f"{safe_id}.json"


def _chat_key(chat_id: str) -> str:
    safe_id = "".join(ch for ch in chat_id if ch.isalnum() or ch in {"-", "_"})[:64]
    return f"streamlit:orca_chat:{safe_id}"


def _redis_client():
    try:
        from redis import Redis
    except ImportError:
        return None
    try:
        client = Redis.from_url(CHAT_REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception:  # noqa: BLE001
        return None


def _load_chat(chat_id: str) -> dict:
    client = _redis_client()
    if client is not None:
        try:
            raw = client.get(_chat_key(chat_id))
            if raw:
                data = json.loads(raw)
                return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            pass
    path = _chat_path(chat_id)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# ── Jobs ──────────────────────────────────────────────────────────────────────

def pending_jobs() -> list[dict]:
    if st.session_state.pending_orca_jobs is None:
        st.session_state.pending_orca_jobs = _load_jobs_from_query()
    return st.session_state.pending_orca_jobs


def add_job(job_dict: dict) -> None:
    pending_jobs().append(job_dict)
    sync_jobs_to_query()


def remove_job(job_id: str) -> None:
    st.session_state.pending_orca_jobs = [
        j for j in pending_jobs() if j.get("job_id") != job_id
    ]
    sync_jobs_to_query()


def sync_jobs_to_query() -> None:
    jobs = pending_jobs()
    if not jobs:
        st.query_params.pop("orca_jobs", None)
        return
    compact = [{"job_id": j.get("job_id"), "kind": j.get("kind"), "symbol": j.get("symbol"),
                "created_at": j.get("created_at"), "started_at": j.get("started_at"),
                "completed_at": j.get("completed_at"), "updated_at": j.get("updated_at"),
                "status": j.get("status")} for j in jobs]
    encoded = base64.urlsafe_b64encode(
        json.dumps(compact, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    st.query_params["orca_jobs"] = encoded


def _load_jobs_from_query() -> list[dict]:
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
    return [j for j in jobs if isinstance(j, dict) and j.get("job_id")]


# ── Time helpers ──────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def fmt_time(value: str | None) -> str:
    parsed = parse_iso(value)
    return parsed.strftime("%H:%M:%S") if parsed else "—"


def fmt_elapsed(start: str | None) -> str:
    s = parse_iso(start)
    if not s:
        return "—"
    secs = max(0, int((utc_now() - s).total_seconds()))
    m, s2 = divmod(secs, 60)
    return f"{m}m {s2}s" if m else f"{s2}s"


def fmt_duration(job: dict) -> str:
    start = parse_iso(job.get("started_at") or job.get("created_at"))
    if not start:
        return "—"
    status = display_status(job)
    end = parse_iso(job.get("completed_at"))
    if not end and status in {"completed", "failed"}:
        end = parse_iso(job.get("updated_at"))
    if not end:
        end = utc_now()
    secs = max(0, int((end - start).total_seconds()))
    m, s2 = divmod(secs, 60)
    return f"{m}m {s2}s" if m else f"{s2}s"


def is_stale(job: dict) -> bool:
    if job.get("status") == "stale":
        return True
    created = parse_iso(job.get("created_at"))
    return bool(
        created
        and job.get("status") in {"queued", "running"}
        and utc_now() - created > timedelta(hours=1)
    )


def display_status(job: dict) -> str:
    status = job.get("status", "unknown")
    if status in {"succeeded", "success"}:
        status = "completed"
    return "stale" if is_stale(job) else status
