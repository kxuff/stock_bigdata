from __future__ import annotations

from typing import Any

from fastapi import status
from pydantic import ValidationError

from app.application.use_cases.advisory_decision_service import DecisionValidationError
from app.bootstrap.container import build_decision_service, build_tool_result_provider
from app.config import load_settings
from app.infrastructure.storage.decision_job_store import PostgresDecisionJobStore
from app.schemas.decision import ErrorResponse
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultValidationError


def run_decision_job(job_id: str, request_payload: dict[str, Any]) -> None:
    settings = load_settings()
    if not settings.decision_job_database_url:
        raise RuntimeError("ORCA_DECISION_JOB_DATABASE_URL is required for worker jobs")
    store = PostgresDecisionJobStore(settings.decision_job_database_url, table_name=settings.decision_job_table)
    request = AdvisoryDecisionRequest.model_validate(request_payload)
    decision_service = build_decision_service(settings)
    tool_result_provider = build_tool_result_provider(settings)

    _update_job(store, job_id, status="running", progress=10, progress_stage="running", started_at=_now_iso())
    try:
        tool_results = tool_result_provider.get_tool_results(request)
        _update_job(store, job_id, progress=50, progress_stage="decision_running")
        decision = decision_service.decide(request, tool_results)
        result = decision.model_dump(mode="json") if hasattr(decision, "model_dump") else decision
        _update_job(
            store,
            job_id,
            status="succeeded",
            progress=100,
            progress_stage="completed",
            run_id=result.get("run_id") if isinstance(result, dict) else None,
            result=result,
            completed_at=_now_iso(),
        )
    except ToolResultValidationError as exc:
        _fail_job(
            store,
            job_id,
            request_id=request.request_id,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="MISSING_REQUIRED_TOOL_RESULT",
            message=str(exc),
            recoverable=True,
            missing_tool_results=_missing_tool_results(str(exc)),
        )
    except TimeoutError as exc:
        _fail_job(
            store,
            job_id,
            request_id=request.request_id,
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            error_code="AGENT_TIMEOUT",
            message=str(exc) or "agent execution timed out",
            recoverable=True,
        )
    except (DecisionValidationError, ValidationError) as exc:
        _fail_job(
            store,
            job_id,
            request_id=request.request_id,
            status_code=422,
            error_code="VALIDATION_FAILED",
            message=str(exc),
            recoverable=False,
        )
    except Exception as exc:  # noqa: BLE001 - worker must persist failure for result endpoint.
        _fail_job(
            store,
            job_id,
            request_id=request.request_id,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_ERROR",
            message=str(exc) or "decision job failed",
            recoverable=True,
        )


def _update_job(store: PostgresDecisionJobStore, job_id: str, **updates: Any) -> None:
    updates["updated_at"] = _now_iso()
    store.update_job(job_id, **updates)


def _fail_job(
    store: PostgresDecisionJobStore,
    job_id: str,
    *,
    request_id: str,
    status_code: int,
    error_code: str,
    message: str,
    recoverable: bool,
    missing_tool_results: list[str] | None = None,
) -> None:
    body = ErrorResponse(
        request_id=request_id,
        status="ERROR",
        error_code=error_code,
        message=message,
        recoverable=recoverable,
        missing_tool_results=missing_tool_results or [],
    ).model_dump(mode="json")
    _update_job(
        store,
        job_id,
        status="failed",
        progress=100,
        progress_stage="failed",
        error={"status_code": status_code, "body": body},
        completed_at=_now_iso(),
    )


def _missing_tool_results(message: str) -> list[str]:
    known_tools = [
        "market_features",
        "risk_snapshot",
        "portfolio_snapshot",
        "sentiment_snapshot",
        "valuation_snapshot",
    ]
    return [tool for tool in known_tools if tool in message]


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
