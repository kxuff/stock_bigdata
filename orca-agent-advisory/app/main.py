from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.schemas.decision import ErrorResponse, PortfolioDecision, SingleSymbolDecision
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultValidationError
from app.config import load_settings
from app.application.ports.tool_result_provider import ToolResultProvider
from app.application.use_cases.advisory_decision_service import AdvisoryDecisionService, DecisionValidationError
from app.bootstrap.container import build_decision_service, build_tool_result_provider
from app.infrastructure.storage.decision_job_store import (
    DecisionJobStore,
    IdempotencyConflictError,
    PostgresDecisionJobStore,
)
from app.infrastructure.queue.decision_job_queue import DecisionJobQueue


app = FastAPI(title="Orca Agent Advisory API", version="0.1.0")

# Dev/first-cut job store only. In-memory dict is not multi-worker safe.
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = Lock()
_job_store: DecisionJobStore | None = None
_job_store_lock = Lock()
_job_queue: DecisionJobQueue | None = None
_job_queue_lock = Lock()


def get_decision_service() -> AdvisoryDecisionService:
    return build_decision_service(load_settings())


def get_tool_result_provider() -> ToolResultProvider:
    return build_tool_result_provider(load_settings())


def get_decision_job_store() -> DecisionJobStore | None:
    global _job_store
    settings = load_settings()
    if not settings.decision_job_database_url:
        return None
    with _job_store_lock:
        if _job_store is None:
            _job_store = PostgresDecisionJobStore(
                settings.decision_job_database_url,
                table_name=settings.decision_job_table,
            )
        return _job_store


def get_decision_job_queue() -> DecisionJobQueue | None:
    global _job_queue
    settings = load_settings()
    if not settings.redis_url:
        return None
    with _job_queue_lock:
        if _job_queue is None:
            _job_queue = DecisionJobQueue(settings)
        return _job_queue


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    load_settings()
    store = get_decision_job_store()
    if store is not None and hasattr(store, "ping"):
        store.ping()
    queue = get_decision_job_queue()
    if queue is not None:
        queue.ping()
    return {"status": "ready"}


@app.get("/api/v1/status")
def api_status() -> dict[str, Any]:
    return {
        "service": "orca-agent-advisory",
        "status": "ready",
        "version": app.version,
        "time": _now_iso(),
    }


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return _error_response(
        request_id=_request_id_from_body(await _safe_json(request)),
        status_code=422,
        error_code="INVALID_REQUEST",
        message=str(exc),
        recoverable=True,
    )


@app.post(
    "/api/v1/advisory/decision",
    response_model=SingleSymbolDecision | PortfolioDecision | ErrorResponse,
)
def create_advisory_decision(
    request: AdvisoryDecisionRequest,
    decision_service: AdvisoryDecisionService = Depends(get_decision_service),
    tool_result_provider: ToolResultProvider = Depends(get_tool_result_provider),
) -> SingleSymbolDecision | PortfolioDecision | JSONResponse:
    try:
        tool_results = tool_result_provider.get_tool_results(request)
        return decision_service.decide(request, tool_results)
    except ToolResultValidationError as exc:
        return _error_response(
            request_id=request.request_id,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="MISSING_REQUIRED_TOOL_RESULT",
            message=str(exc),
            recoverable=True,
            missing_tool_results=_missing_tool_results(str(exc)),
        )
    except TimeoutError as exc:
        return _error_response(
            request_id=request.request_id,
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            error_code="AGENT_TIMEOUT",
            message=str(exc) or "agent execution timed out",
            recoverable=True,
        )
    except (DecisionValidationError, ValidationError) as exc:
        return _error_response(
            request_id=request.request_id,
            status_code=422,
            error_code="VALIDATION_FAILED",
            message=str(exc),
            recoverable=False,
        )


@app.post("/api/v1/advisory/decision-jobs", status_code=status.HTTP_202_ACCEPTED)
def create_advisory_decision_job(
    request: AdvisoryDecisionRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    job_id = str(uuid4())
    now = _now_iso()
    job = {
        "job_id": job_id,
        "request_id": request.request_id,
        "status": "queued",
        "progress": 0,
        "progress_stage": "queued",
        "run_id": None,
        "error": None,
        "result": None,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
    }
    settings = load_settings()
    store = get_decision_job_store()
    queue = get_decision_job_queue()
    if queue is not None and store is None:
        return _error_response(
            request_id=request.request_id,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="JOB_STORE_REQUIRED",
            message="ORCA_DECISION_JOB_DATABASE_URL is required when ORCA_REDIS_URL is configured",
            recoverable=True,
        )
    if store is not None:
        try:
            result = store.create_job(
                job,
                request_payload=request.model_dump(mode="json"),
                idempotency_key=http_request.headers.get("Idempotency-Key"),
                tenant_id=http_request.headers.get("X-Tenant-Id", "local"),
                user_id=http_request.headers.get("X-User-Id"),
                created_by=http_request.headers.get("X-User-Id"),
            )
        except IdempotencyConflictError as exc:
            return _error_response(
                request_id=request.request_id,
                status_code=status.HTTP_409_CONFLICT,
                error_code="IDEMPOTENCY_CONFLICT",
                message=str(exc),
                recoverable=False,
            )
        job = result.job
        if not result.created:
            return _job_public(job, include_links=True)
    else:
        with _jobs_lock:
            _jobs[job_id] = job
    if queue is not None and store is not None:
        try:
            queue.enqueue_decision_job(
                job_id,
                request.model_dump(mode="json"),
                timeout_seconds=settings.agent_timeout_seconds + 120,
            )
        except Exception as exc:  # noqa: BLE001 - queued job must not stay queued if enqueue fails.
            _fail_job(
                job_id,
                request_id=request.request_id,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                error_code="QUEUE_UNAVAILABLE",
                message=str(exc) or "decision job queue unavailable",
                recoverable=True,
            )
            return _error_response(
                request_id=request.request_id,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                error_code="QUEUE_UNAVAILABLE",
                message="decision job queue unavailable",
                recoverable=True,
            )
    else:
        decision_service = get_decision_service()
        tool_result_provider = get_tool_result_provider()
        background_tasks.add_task(_run_decision_job, job_id, request, decision_service, tool_result_provider)
    return _job_public(job, include_links=True)


@app.get("/api/v1/advisory/decision-jobs/{job_id}", response_model=None)
def get_advisory_decision_job(job_id: str) -> JSONResponse | dict[str, Any]:
    job = _get_job(job_id)
    if job is None:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": "job not found"})
    return _job_public(job)


@app.get("/api/v1/advisory/decision-jobs/{job_id}/result", response_model=None)
def get_advisory_decision_job_result(job_id: str) -> JSONResponse | dict[str, Any]:
    job = _get_job(job_id)
    if job is None:
        return _error_response(
            request_id="UNKNOWN",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="JOB_NOT_FOUND",
            message="job not found",
            recoverable=False,
        )
    if job["status"] == "failed":
        error = job["error"]
        if isinstance(error, dict) and "status_code" in error and "body" in error:
            return JSONResponse(status_code=error["status_code"], content=error["body"])
        return _error_response(
            request_id=job.get("request_id") or "UNKNOWN",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_ERROR",
            message=str(error) or "decision job failed",
            recoverable=True,
        )
    if job["status"] != "succeeded":
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=_job_public(job))
    return job["result"]


@app.get("/api/v1/data/readiness")
def data_readiness(
    symbols: str = Query(..., min_length=1),
    decision_mode: str = Query("single_symbol_advisory"),
    tool_result_provider: ToolResultProvider = Depends(get_tool_result_provider),
) -> dict[str, Any]:
    symbol_list = [symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()]
    request_symbol = symbol_list[0] if decision_mode == "single_symbol_advisory" and symbol_list else "UNKNOWN"
    now = datetime.now(UTC)
    request = AdvisoryDecisionRequest(
        request_id=f"readiness-{uuid4()}",
        timestamp=now,
        as_of_timestamp=now,
        user_query="data readiness check",
        decision_mode=decision_mode,
        symbols=[request_symbol] if decision_mode == "single_symbol_advisory" else symbol_list,
    )
    try:
        bundle = tool_result_provider.get_tool_results(request)
    except Exception as exc:  # noqa: BLE001 - readiness must report provider failure, not run CrewAI.
        return {"ready": False, "symbols": symbol_list, "decision_mode": decision_mode, "error": str(exc), "tools": {}}

    tools = _readiness_tools(bundle, symbol_list)
    required_tools = ["market_features"]
    if decision_mode == "portfolio_recommendation":
        required_tools.extend(["risk_snapshot", "portfolio_snapshot"])
    ready = all(
        tools[tool]["status"] == "SUCCESS"
        and not tools[tool]["freshness"].get("is_stale", True)
        and not tools[tool]["missing_symbols"]
        for tool in required_tools
    )
    return {"ready": ready, "symbols": symbol_list, "decision_mode": decision_mode, "tools": tools}


def _error_response(
    *,
    request_id: str,
    status_code: int,
    error_code: str,
    message: str,
    recoverable: bool,
    missing_tool_results: list[str] | None = None,
) -> JSONResponse:
    response = ErrorResponse(
        request_id=request_id,
        status="ERROR",
        error_code=error_code,
        message=message,
        recoverable=recoverable,
        missing_tool_results=missing_tool_results or [],
    )
    return JSONResponse(status_code=status_code, content=response.model_dump(mode="json"))


def _missing_tool_results(message: str) -> list[str]:
    known_tools = [
        "market_features",
        "risk_snapshot",
        "portfolio_snapshot",
        "sentiment_snapshot",
        "valuation_snapshot",
    ]
    return [tool for tool in known_tools if tool in message]


def _run_decision_job(
    job_id: str,
    request: AdvisoryDecisionRequest,
    decision_service: AdvisoryDecisionService,
    tool_result_provider: ToolResultProvider,
) -> None:
    _update_job(job_id, status="running", progress=10, progress_stage="running", started_at=_now_iso())
    try:
        tool_results = tool_result_provider.get_tool_results(request)
        _update_job(job_id, progress=50, progress_stage="decision_running")
        decision = decision_service.decide(request, tool_results)
        result = decision.model_dump(mode="json") if hasattr(decision, "model_dump") else decision
        _update_job(
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
            job_id,
            request_id=request.request_id,
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            error_code="AGENT_TIMEOUT",
            message=str(exc) or "agent execution timed out",
            recoverable=True,
        )
    except (DecisionValidationError, ValidationError) as exc:
        _fail_job(
            job_id,
            request_id=request.request_id,
            status_code=422,
            error_code="VALIDATION_FAILED",
            message=str(exc),
            recoverable=False,
        )
    except Exception as exc:  # noqa: BLE001 - job surface stores failure for result endpoint.
        _fail_job(
            job_id,
            request_id=request.request_id,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_ERROR",
            message=str(exc) or "decision job failed",
            recoverable=True,
        )


def _fail_job(
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
        job_id,
        status="failed",
        progress=100,
        progress_stage="failed",
        error={"status_code": status_code, "body": body},
        completed_at=_now_iso(),
    )


def _update_job(job_id: str, **updates: Any) -> None:
    updates["updated_at"] = _now_iso()
    store = get_decision_job_store()
    if store is not None:
        store.update_job(job_id, **updates)
        return
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(updates)


def _get_job(job_id: str) -> dict[str, Any] | None:
    store = get_decision_job_store()
    if store is not None:
        return store.get_job(job_id)
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _job_public(job: dict[str, Any], *, include_links: bool = False) -> dict[str, Any]:
    public_fields = {
        "job_id",
        "request_id",
        "status",
        "progress",
        "progress_stage",
        "run_id",
        "error",
        "created_at",
        "updated_at",
        "started_at",
        "completed_at",
    }
    payload = {key: value for key, value in job.items() if key in public_fields}
    if include_links:
        payload["links"] = {
            "status": f"/api/v1/advisory/decision-jobs/{job['job_id']}",
            "result": f"/api/v1/advisory/decision-jobs/{job['job_id']}/result",
        }
    return payload


def _readiness_tools(bundle: Any, symbols: list[str]) -> dict[str, Any]:
    tool_names = [
        "market_features",
        "ml_predictions",
        "sentiment_snapshot",
        "valuation_snapshot",
        "risk_snapshot",
        "portfolio_snapshot",
    ]
    tools: dict[str, Any] = {}
    for tool_name in tool_names:
        result = getattr(bundle, tool_name, None)
        if result is None:
            tools[tool_name] = {"status": "MISSING", "freshness": {}, "missing_symbols": symbols, "error": None}
            continue
        data = getattr(result, "data", None)
        if isinstance(data, dict):
            present = set(data.keys())
        elif tool_name == "portfolio_snapshot" and data is not None:
            present = {holding.symbol for holding in getattr(data, "holdings", [])}
        else:
            present = set()
        tools[tool_name] = {
            "status": str(result.status),
            "freshness": result.freshness.model_dump(mode="json"),
            "missing_symbols": [symbol for symbol in symbols if symbol not in present],
            "error": result.error_message,
        }
    return tools


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def _safe_json(request: Request) -> Any:
    try:
        return await request.json()
    except ValueError:
        return None


def _request_id_from_body(body: Any) -> str:
    if isinstance(body, dict) and isinstance(body.get("request_id"), str):
        return body["request_id"]
    return "UNKNOWN"
