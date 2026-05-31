from __future__ import annotations

import os
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


def fetch_readiness(symbols: list[str], decision_mode: str = "single_symbol_advisory", timeout: float = 10.0) -> dict[str, Any]:
    response = requests.get(
        api_url("/api/v1/data/readiness"),
        params={"symbols": ",".join(symbols), "decision_mode": decision_mode},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def create_decision_job(payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    response = requests.post(api_url("/api/v1/advisory/decision-jobs"), json=payload, timeout=timeout)
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
