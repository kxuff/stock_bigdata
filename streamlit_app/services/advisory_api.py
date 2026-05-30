from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_ADVISORY_URL = "http://127.0.0.1:8000/api/v1/advisory/decision"


def advisory_url() -> str:
    return os.getenv("ORCA_API_URL", DEFAULT_ADVISORY_URL)


def fetch_advisory_decision(payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    """Call ORCA advisory API. UI pages stay mock-first and do not call this by default."""
    response = requests.post(advisory_url(), json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()
