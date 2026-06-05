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


def advisory_url() -> str:
    return api_url("/api/v1/advisory/decision")


def fetch_advisory_decision(payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    response = requests.post(advisory_url(), json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_health(timeout: float = 3.0) -> dict[str, Any]:
    response = requests.get(api_url("/healthz"), timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_status(timeout: float = 3.0) -> dict[str, Any]:
    response = requests.get(api_url("/api/v1/status"), timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_readiness(symbols: list[str], decision_mode: str = "single_symbol_advisory", timeout: float = 60.0) -> dict[str, Any]:
    response = requests.get(
        api_url("/api/v1/data/readiness"),
        params={"symbols": ",".join(symbols), "decision_mode": decision_mode},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def fetch_data_coverage(symbols: list[str], decision_mode: str = "single_symbol_advisory", timeout: float = 10.0) -> dict[str, Any]:
    response = requests.get(
        api_url("/api/v1/data/coverage"),
        params={"symbols": ",".join(symbols), "decision_mode": decision_mode},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def fetch_advisory_picks(
    limit: int = 25,
    min_pred_a: float = 0.06,
    max_risk_prob: float = 0.3,
    as_of_date: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "limit": limit,
        "min_pred_a": min_pred_a,
        "max_risk_prob": max_risk_prob,
    }
    if as_of_date:
        params["as_of_date"] = as_of_date
    response = requests.get(api_url("/api/v1/advisory/picks"), params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_advisory_pick_detail(symbol: str, timeout: float = 10.0) -> dict[str, Any]:
    response = requests.get(api_url(f"/api/v1/advisory/picks/{symbol}"), timeout=timeout)
    response.raise_for_status()
    return response.json()


def create_decision_job(payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    response = requests.post(api_url("/api/v1/advisory/decision-jobs"), json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def create_agent_query(payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    response = requests.post(api_url("/api/v1/agent/query"), json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def create_agent_query_job(payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    response = requests.post(api_url("/api/v1/agent/query-jobs"), json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_decision_job(job_id: str, timeout: float = 5.0) -> dict[str, Any]:
    response = requests.get(api_url(f"/api/v1/advisory/decision-jobs/{job_id}"), timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_decision_job_result(job_id: str, timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
    response = requests.get(api_url(f"/api/v1/advisory/decision-jobs/{job_id}/result"), timeout=timeout)
    if response.status_code not in {200, 202}:
        response.raise_for_status()
    return response.status_code, response.json()


def get_agent_query_job_result(job_id: str, timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
    response = requests.get(api_url(f"/api/v1/agent/query-jobs/{job_id}/result"), timeout=timeout)
    if response.status_code not in {200, 202}:
        response.raise_for_status()
    return response.status_code, response.json()


def stream_decision_job_events(job_id: str, timeout: tuple[float, float | None] = (5.0, None)):
    response = requests.get(
        api_url(f"/api/v1/advisory/decision-jobs/{job_id}/events"),
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
