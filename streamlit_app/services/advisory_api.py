from __future__ import annotations

import os
import json
from typing import Any

import requests


DEFAULT_ORCA_API_BASE_URL = "http://127.0.0.1:8000"


def api_base_url() -> str:
    legacy_url = os.getenv("ORCA_API_URL")
    if legacy_url:
        return legacy_url.rstrip("/").removesuffix("/api/v1/advisory/decision").rstrip("/")
    return os.getenv("ORCA_API_BASE_URL", DEFAULT_ORCA_API_BASE_URL).rstrip("/")


def api_url(path: str) -> str:
    return f"{api_base_url()}/{path.lstrip('/')}"


def fetch_health(timeout: float = 3.0) -> dict[str, Any]:
    response = requests.get(api_url("/healthz"), timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_status(timeout: float = 3.0) -> dict[str, Any]:
    response = requests.get(api_url("/api/v1/status"), timeout=timeout)
    response.raise_for_status()
    return response.json()


def create_agent_query_job(payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    response = requests.post(api_url("/api/v1/agent/query-jobs"), json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def stream_agent_query_job_events(job_id: str, timeout: tuple[float, float | None] = (5.0, None)):
    response = requests.get(
        api_url(f"/api/v1/agent/query-jobs/{job_id}/events"),
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()
    event = "message"
    data_lines: list[str] = []
    with response:
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.strip()
            if not line:
                if data_lines:
                    data = "\n".join(data_lines)
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        payload = {"raw": data}
                    yield {"event": event, "data": payload}
                event = "message"
                data_lines = []
                continue
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
