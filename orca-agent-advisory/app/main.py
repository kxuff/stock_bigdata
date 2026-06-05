import asyncio
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from app.schemas.decision import ErrorResponse, PortfolioDecision, SingleSymbolDecision
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.agent import AgentQueryRequest, AgentQueryResponse
from app.schemas.tool_results import ToolResultValidationError
from app.config import load_settings
from app.application.ports.market_screen_provider import MarketScreenProvider
from app.application.ports.tool_result_provider import ToolResultProvider
from app.application.use_cases.advisory_decision_service import AdvisoryDecisionService, DecisionValidationError
from app.application.use_cases.autonomous_agent_service import AutonomousAgentService
from app.bootstrap.container import (
    build_autonomous_agent_service,
    build_decision_service,
    build_market_screen_provider,
    build_tool_result_provider,
)
from app.infrastructure.storage.decision_job_store import (
    DecisionJobStore,
    IdempotencyConflictError,
    PostgresDecisionJobStore,
)
from app.infrastructure.storage.agent_route_audit_store import (
    AgentRouteAuditEntry,
    AgentRouteAuditStore,
    NoopAgentRouteAuditStore,
    PostgresAgentRouteAuditStore,
)
from app.infrastructure.queue.decision_job_queue import DecisionJobQueue


app = FastAPI(title="Orca Agent Advisory API", version="0.1.0")
logger = logging.getLogger(__name__)


# Dev/first-cut job store only. In-memory dict is not multi-worker safe.
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = Lock()
_job_store: DecisionJobStore | None = None
_job_store_lock = Lock()
_job_queue: DecisionJobQueue | None = None
_job_queue_lock = Lock()
_agent_route_audit_store: AgentRouteAuditStore | None = None
_agent_route_audit_store_lock = Lock()


def get_decision_service() -> AdvisoryDecisionService:
    return build_decision_service(load_settings())


def get_tool_result_provider() -> ToolResultProvider:
    return build_tool_result_provider(load_settings())


def get_market_screen_provider() -> MarketScreenProvider:
    return build_market_screen_provider(load_settings())


def get_autonomous_agent_service() -> AutonomousAgentService:
    return build_autonomous_agent_service(load_settings())


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


def get_agent_route_audit_store() -> AgentRouteAuditStore:
    global _agent_route_audit_store
    settings = load_settings()
    if not settings.agent_route_audit_database_url:
        return NoopAgentRouteAuditStore()
    with _agent_route_audit_store_lock:
        if _agent_route_audit_store is None:
            try:
                _agent_route_audit_store = PostgresAgentRouteAuditStore(
                    settings.agent_route_audit_database_url,
                    table_name=settings.agent_route_audit_table,
                )
            except Exception as exc:  # noqa: BLE001 - audit must never block traffic.
                logger.warning("agent route audit store unavailable: %s", exc)
                _agent_route_audit_store = NoopAgentRouteAuditStore()
        return _agent_route_audit_store


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
    return _create_decision_job(request, http_request, background_tasks)


@app.post("/api/v1/agent/query", response_model=AgentQueryResponse)
def create_agent_query(
    request: AgentQueryRequest,
    http_request: Request,
    autonomous_agent_service: AutonomousAgentService = Depends(get_autonomous_agent_service),
    audit_store: AgentRouteAuditStore = Depends(get_agent_route_audit_store),
) -> AgentQueryResponse:
    started = time.perf_counter()
    try:
        response = autonomous_agent_service.query(request)
        _record_agent_route_audit(
            audit_store,
            request=request,
            http_request=http_request,
            response=response,
            status="succeeded",
            latency_ms=_latency_ms(started),
        )
        return response
    except ToolResultValidationError as exc:
        _record_agent_route_audit(
            audit_store,
            request=request,
            http_request=http_request,
            status="failed",
            error_code="MISSING_REQUIRED_TOOL_RESULT",
            latency_ms=_latency_ms(started),
        )
        return _error_response(
            request_id=_agent_request_id(request) or "UNKNOWN",
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="MISSING_REQUIRED_TOOL_RESULT",
            message=str(exc),
            recoverable=True,
            missing_tool_results=_missing_tool_results(str(exc)),
        )
    except Exception as exc:
        _record_agent_route_audit(
            audit_store,
            request=request,
            http_request=http_request,
            status="failed",
            error_code=_agent_error_code(exc),
            latency_ms=_latency_ms(started),
        )
        raise


@app.post("/api/v1/agent/query-jobs", status_code=status.HTTP_202_ACCEPTED)
def create_agent_query_job(
    request: AgentQueryRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    autonomous_agent_service: AutonomousAgentService = Depends(get_autonomous_agent_service),
    audit_store: AgentRouteAuditStore = Depends(get_agent_route_audit_store),
) -> dict[str, Any]:
    job_id = str(uuid4())
    now = _now_iso()
    job = {
        "job_id": job_id,
        "request_id": "UNKNOWN",
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
    with _jobs_lock:
        _jobs[job_id] = job
    background_tasks.add_task(
        _run_agent_query_job,
        job_id,
        request,
        autonomous_agent_service,
        audit_store,
        http_request.headers.get("X-Tenant-Id"),
        http_request.headers.get("X-User-Id"),
    )
    return _job_public(job, include_links=True, base_path="/api/v1/agent/query-jobs")


@app.get("/api/v1/agent/query-jobs/{job_id}", response_model=None)
def get_agent_query_job(job_id: str) -> JSONResponse | dict[str, Any]:
    job = _get_job(job_id)
    if job is None:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": "job not found"})
    return _job_public(job, base_path="/api/v1/agent/query-jobs")


@app.get("/api/v1/agent/query-jobs/{job_id}/result", response_model=None)
def get_agent_query_job_result(job_id: str) -> JSONResponse | dict[str, Any]:
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
            message=str(error) or "agent query job failed",
            recoverable=True,
        )
    if job["status"] != "succeeded":
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=_job_public(job, base_path="/api/v1/agent/query-jobs"))
    return job["result"]


@app.get("/api/v1/agent/query-jobs/{job_id}/events", response_model=None)
async def stream_agent_query_job_events(job_id: str) -> StreamingResponse:
    async def event_stream():
        previous_payload = None
        while True:
            job = _get_job(job_id)
            if job is None:
                yield _sse_event("error", {"error_code": "JOB_NOT_FOUND", "message": "job not found"})
                return

            public_job = _job_public(job, base_path="/api/v1/agent/query-jobs")
            payload = json.dumps(public_job, separators=(",", ":"), sort_keys=True)
            if payload != previous_payload:
                yield _sse_event("status", public_job)
                previous_payload = payload
            else:
                yield _sse_event("heartbeat", {"job_id": job_id, "time": _now_iso()})

            if job.get("status") == "succeeded":
                yield _sse_event("result", job.get("result") or {})
                return
            if job.get("status") == "failed":
                yield _sse_event("failure", job.get("error") or {})
                return

            await asyncio.sleep(2)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _create_decision_job(
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


@app.get("/api/v1/advisory/decision-jobs/{job_id}/events", response_model=None)
async def stream_advisory_decision_job_events(job_id: str) -> StreamingResponse:
    async def event_stream():
        previous_payload = None
        while True:
            job = _get_job(job_id)
            if job is None:
                yield _sse_event("error", {"error_code": "JOB_NOT_FOUND", "message": "job not found"})
                return

            public_job = _job_public(job)
            payload = json.dumps(public_job, separators=(",", ":"), sort_keys=True)
            if payload != previous_payload:
                yield _sse_event("status", public_job)
                previous_payload = payload
            else:
                yield _sse_event("heartbeat", {"job_id": job_id, "time": _now_iso()})

            if job.get("status") == "succeeded":
                yield _sse_event("result", job.get("result") or {})
                return
            if job.get("status") == "failed":
                yield _sse_event("failure", job.get("error") or {})
                return

            await asyncio.sleep(2)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


@app.get("/api/v1/data/coverage")
def data_coverage(
    symbols: str = Query(..., min_length=1),
    decision_mode: str = Query("single_symbol_advisory"),
    tool_result_provider: ToolResultProvider = Depends(get_tool_result_provider),
) -> dict[str, Any]:
    symbol_list = _query_symbols(symbols)
    if not symbol_list:
        return {
            "ready": False,
            "symbols": [],
            "decision_mode": decision_mode,
            "rows": [],
            "error": "No valid symbols were provided.",
        }

    now = datetime.now(UTC)
    request_mode = decision_mode if len(symbol_list) == 1 else "portfolio_recommendation"
    request = AdvisoryDecisionRequest(
        request_id=f"coverage-{uuid4()}",
        timestamp=now,
        as_of_timestamp=now,
        user_query="data coverage check",
        decision_mode=request_mode,
        symbols=symbol_list,
    )
    try:
        bundle = tool_result_provider.get_tool_results(request)
    except Exception as exc:  # noqa: BLE001 - coverage must fail soft for UI.
        return {
            "ready": False,
            "symbols": symbol_list,
            "decision_mode": decision_mode,
            "rows": [
                {
                    "symbol": symbol,
                    "ready": False,
                    "latest_timestamp": None,
                    "tools": {},
                    "warnings": [str(exc)],
                }
                for symbol in symbol_list
            ],
            "error": str(exc),
        }

    rows = [_coverage_row(bundle, symbol) for symbol in symbol_list]
    return {
        "ready": all(row["ready"] for row in rows),
        "symbols": symbol_list,
        "decision_mode": decision_mode,
        "rows": rows,
    }


@app.get("/api/v1/advisory/picks")
def advisory_picks(
    limit: int = Query(25, ge=1, le=100),
    min_pred_a: float = Query(0.06, ge=-1.0, le=1.0),
    max_risk_prob: float = Query(0.3, ge=0.0, le=1.0),
    as_of_date: str | None = Query(None),
    market_screen_provider: MarketScreenProvider = Depends(get_market_screen_provider),
) -> dict[str, Any]:
    try:
        rows = _screen_rows(market_screen_provider, max(limit * 4, limit), as_of_date)
    except Exception as exc:  # noqa: BLE001 - UI needs structured failure.
        return {
            "data": [],
            "count": 0,
            "limit": limit,
            "filters": {"min_pred_a": min_pred_a, "max_risk_prob": max_risk_prob, "as_of_date": as_of_date},
            "warnings": [str(exc)],
        }

    picks = [_pick_from_row(row) for row in rows]
    picks = [
        pick
        for pick in picks
        if pick["pred_a"] is not None
        and pick["risk_prob"] is not None
        and pick["pred_a"] >= min_pred_a
        and pick["risk_prob"] <= max_risk_prob
    ]
    picks = sorted(picks, key=lambda pick: (pick["final_score"] is not None, pick["final_score"] or -9999), reverse=True)
    data = picks[:limit]
    warnings = [] if data else ["No prediction rows matched the requested filters."]
    return {
        "data": data,
        "count": len(data),
        "limit": limit,
        "filters": {"min_pred_a": min_pred_a, "max_risk_prob": max_risk_prob, "as_of_date": as_of_date},
        "warnings": warnings,
    }


@app.get("/api/v1/advisory/picks/{symbol}", response_model=None)
def advisory_pick_detail(
    symbol: str,
    market_screen_provider: MarketScreenProvider = Depends(get_market_screen_provider),
) -> dict[str, Any] | JSONResponse:
    normalized = _normalize_symbol(symbol)
    if not normalized:
        return _error_response(
            request_id="picks-UNKNOWN",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="PICK_NOT_FOUND",
            message="pick not found",
            recoverable=True,
        )
    try:
        rows = market_screen_provider.load_symbols([normalized])
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            request_id=f"picks-{normalized}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="PICK_PROVIDER_UNAVAILABLE",
            message=str(exc) or "pick provider unavailable",
            recoverable=True,
        )
    if not rows:
        return _error_response(
            request_id=f"picks-{normalized}",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="PICK_NOT_FOUND",
            message=f"no pick found for {normalized}",
            recoverable=True,
        )
    pick = _pick_from_row(rows[0])
    if not pick["symbol"]:
        pick["symbol"] = normalized
    return pick


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


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'), default=str)}\n\n"


def _missing_tool_results(message: str) -> list[str]:
    known_tools = [
        "market_features",
        "risk_snapshot",
        "portfolio_snapshot",
        "sentiment_snapshot",
        "valuation_snapshot",
    ]
    return [tool for tool in known_tools if tool in message]


def _query_symbols(symbols: str) -> list[str]:
    normalized: list[str] = []
    for value in symbols.split(","):
        symbol = _normalize_symbol(value)
        if symbol and symbol not in normalized:
            normalized.append(symbol)
    return normalized


def _normalize_symbol(value: Any) -> str | None:
    symbol = str(value or "").strip().upper().replace(".", "-")
    return symbol or None


def _coverage_row(bundle: Any, symbol: str) -> dict[str, Any]:
    tool_names = [
        "market_features",
        "ml_predictions",
        "risk_snapshot",
        "sentiment_snapshot",
        "valuation_snapshot",
    ]
    tools = {tool_name: _symbol_tool_coverage(getattr(bundle, tool_name, None), symbol) for tool_name in tool_names}
    required = ["market_features", "ml_predictions", "risk_snapshot"]
    ready = all(tools[tool]["status"] == "SUCCESS" for tool in required)
    warnings = [
        f"{tool} {data['status'].lower()}"
        for tool, data in tools.items()
        if data["status"] != "SUCCESS"
    ]
    latest_timestamp = _latest_timestamp(
        data["freshness"].get("last_updated_at")
        for data in tools.values()
        if data.get("present") and isinstance(data.get("freshness"), dict)
    )
    return {
        "symbol": symbol,
        "ready": ready,
        "latest_timestamp": latest_timestamp,
        "tools": tools,
        "warnings": warnings,
    }


def _symbol_tool_coverage(result: Any, symbol: str) -> dict[str, Any]:
    if result is None:
        return {"status": "MISSING", "present": False, "freshness": {}, "error": None}

    status_value = _status_value(getattr(result, "status", None))
    data = getattr(result, "data", None)
    present = _symbol_present(data, symbol)
    freshness_model = getattr(result, "freshness", None)
    freshness = freshness_model.model_dump(mode="json") if hasattr(freshness_model, "model_dump") else {}

    if present and freshness.get("is_stale"):
        status_for_symbol = "STALE"
    elif present and status_value in {"SUCCESS", "PARTIAL"}:
        status_for_symbol = "SUCCESS"
    elif present:
        status_for_symbol = status_value
    elif status_value in {"UNAVAILABLE", "ERROR"}:
        status_for_symbol = status_value
    else:
        status_for_symbol = "MISSING"

    return {
        "status": status_for_symbol,
        "present": present,
        "freshness": freshness,
        "error": getattr(result, "error_message", None),
    }


def _symbol_present(data: Any, symbol: str) -> bool:
    if isinstance(data, dict):
        return symbol in {str(key).upper() for key in data.keys()}
    holdings = getattr(data, "holdings", None)
    if holdings is not None:
        return symbol in {str(getattr(holding, "symbol", "")).upper() for holding in holdings}
    return False


def _status_value(value: Any) -> str:
    if value is None:
        return "MISSING"
    return str(getattr(value, "value", value)).upper()


def _screen_rows(provider: MarketScreenProvider, limit: int, as_of_date: str | None) -> list[dict[str, Any]]:
    if as_of_date:
        try:
            return provider.screen_latest(limit=limit, as_of_date=as_of_date)  # type: ignore[call-arg]
        except TypeError:
            pass
    return provider.screen_latest(limit=limit)


def _pick_from_row(row: dict[str, Any]) -> dict[str, Any]:
    symbol = _normalize_symbol(_first_present(row, "Symbol", "symbol")) or ""
    pred_a = _float_or_none(_first_present(row, "pred_a", "Pred_A"))
    risk_prob = _risk_prob(_first_present(row, "risk_prob", "Risk_Prob_%", "RiskProb"))
    final_score = _float_or_none(_first_present(row, "final_score", "FinalScore"))
    if final_score is None and pred_a is not None and risk_prob is not None:
        final_score = pred_a * (1 - risk_prob)
    date_value = _first_present(row, "Date", "date", "Datetime", "datetime", "as_of", "freshness")
    latest_price = _float_or_none(_first_present(row, "latest_price", "Close", "close", "entry_price", "Entry_Price"))
    entry_price = _float_or_none(_first_present(row, "entry_price", "Entry_Price", "Close", "close", "latest_price"))
    warnings = _pick_warnings(
        symbol=symbol,
        date_value=date_value,
        entry_price=entry_price,
        pred_a=pred_a,
        risk_prob=risk_prob,
        final_score=final_score,
    )
    return {
        "symbol": symbol,
        "date": _date_string(date_value),
        "entry_price": entry_price,
        "pred_a": pred_a,
        "risk_prob": risk_prob,
        "final_score": final_score,
        "latest_price": latest_price,
        "ready": not warnings,
        "warnings": warnings,
    }


def _pick_warnings(
    *,
    symbol: str,
    date_value: Any,
    entry_price: float | None,
    pred_a: float | None,
    risk_prob: float | None,
    final_score: float | None,
) -> list[str]:
    warnings: list[str] = []
    if not symbol:
        warnings.append("missing symbol")
    if _date_string(date_value) is None:
        warnings.append("missing date")
    if entry_price is None:
        warnings.append("missing entry_price")
    if pred_a is None:
        warnings.append("missing pred_a")
    if risk_prob is None:
        warnings.append("missing risk_prob")
    if final_score is None:
        warnings.append("missing final_score")
    return warnings


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _risk_prob(value: Any) -> float | None:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    return parsed / 100 if parsed > 1 else parsed


def _date_string(value: Any) -> str | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.date().isoformat()
    except ValueError:
        text = str(value).strip()
        return text[:10] if text else None


def _latest_timestamp(values: Any) -> str | None:
    parsed_values = [_parse_timestamp(value) for value in values if value]
    parsed_values = [value for value in parsed_values if value is not None]
    if not parsed_values:
        return None
    return max(parsed_values).isoformat()


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


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


def _run_agent_query_job(
    job_id: str,
    request: AgentQueryRequest,
    autonomous_agent_service: AutonomousAgentService,
    audit_store: AgentRouteAuditStore,
    tenant_id: str | None = None,
    user_id: str | None = None,
) -> None:
    started = time.perf_counter()
    _update_job(job_id, status="running", progress=10, progress_stage="running", started_at=_now_iso())
    try:
        response = autonomous_agent_service.query(request)
        result = response.model_dump(mode="json")
        _update_job(
            job_id,
            status="succeeded",
            progress=100,
            progress_stage="completed",
            result=result,
            completed_at=_now_iso(),
        )
        _record_agent_route_audit(
            audit_store,
            request=request,
            job_id=job_id,
            tenant_id=tenant_id,
            user_id=user_id,
            response=response,
            status="succeeded",
            latency_ms=_latency_ms(started),
        )
    except ToolResultValidationError as exc:
        _fail_job(
            job_id,
            request_id="UNKNOWN",
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="MISSING_REQUIRED_TOOL_RESULT",
            message=str(exc),
            recoverable=True,
            missing_tool_results=_missing_tool_results(str(exc)),
        )
        _record_agent_route_audit(
            audit_store,
            request=request,
            job_id=job_id,
            tenant_id=tenant_id,
            user_id=user_id,
            status="failed",
            error_code="MISSING_REQUIRED_TOOL_RESULT",
            latency_ms=_latency_ms(started),
        )
    except TimeoutError as exc:
        _fail_job(
            job_id,
            request_id="UNKNOWN",
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            error_code="AGENT_TIMEOUT",
            message=str(exc) or "agent execution timed out",
            recoverable=True,
        )
        _record_agent_route_audit(
            audit_store,
            request=request,
            job_id=job_id,
            tenant_id=tenant_id,
            user_id=user_id,
            status="failed",
            error_code="AGENT_TIMEOUT",
            latency_ms=_latency_ms(started),
        )
    except (DecisionValidationError, ValidationError) as exc:
        _fail_job(
            job_id,
            request_id="UNKNOWN",
            status_code=422,
            error_code="VALIDATION_FAILED",
            message=str(exc),
            recoverable=False,
        )
        _record_agent_route_audit(
            audit_store,
            request=request,
            job_id=job_id,
            tenant_id=tenant_id,
            user_id=user_id,
            status="failed",
            error_code="VALIDATION_FAILED",
            latency_ms=_latency_ms(started),
        )
    except Exception as exc:  # noqa: BLE001 - job surface stores failure for result endpoint.
        _fail_job(
            job_id,
            request_id="UNKNOWN",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_ERROR",
            message=str(exc) or "agent query job failed",
            recoverable=True,
        )
        _record_agent_route_audit(
            audit_store,
            request=request,
            job_id=job_id,
            tenant_id=tenant_id,
            user_id=user_id,
            status="failed",
            error_code="INTERNAL_ERROR",
            latency_ms=_latency_ms(started),
        )


def _record_agent_route_audit(
    audit_store: AgentRouteAuditStore,
    *,
    request: AgentQueryRequest,
    http_request: Request | None = None,
    job_id: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
    response: AgentQueryResponse | None = None,
    status: str,
    error_code: str | None = None,
    latency_ms: int | None = None,
) -> None:
    if http_request is not None:
        tenant_id = http_request.headers.get("X-Tenant-Id")
        user_id = http_request.headers.get("X-User-Id")
    route = response.route.value if response is not None else None
    try:
        audit_store.record(
            AgentRouteAuditEntry(
                audit_id=str(uuid4()),
                request_id=_agent_request_id(request),
                job_id=job_id,
                tenant_id=tenant_id,
                user_id=user_id,
                message_hash=hashlib.sha256(request.message.encode("utf-8")).hexdigest(),
                route=route,
                router_confidence=response.router_confidence if response is not None else None,
                symbols=response.symbols if response is not None else list(request.context.symbols),
                status=status,
                error_code=error_code,
                latency_ms=latency_ms,
                created_at=_now_iso(),
            )
        )
    except Exception as exc:  # noqa: BLE001 - audit is best-effort.
        logger.warning("agent route audit failed: %s", exc)


def _agent_request_id(request: AgentQueryRequest) -> str | None:
    value = request.context.metadata.get("request_id")
    return value if isinstance(value, str) else None


def _agent_error_code(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "AGENT_TIMEOUT"
    if isinstance(exc, ToolResultValidationError):
        return "MISSING_REQUIRED_TOOL_RESULT"
    if isinstance(exc, (DecisionValidationError, ValidationError)):
        return "VALIDATION_FAILED"
    return "INTERNAL_ERROR"


def _latency_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


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
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(updates)
            return
    store = get_decision_job_store()
    if store is not None:
        store.update_job(job_id, **updates)
        return


def _get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            return dict(job)
    store = get_decision_job_store()
    if store is not None:
        return store.get_job(job_id)
    return None


def _job_public(
    job: dict[str, Any],
    *,
    include_links: bool = False,
    base_path: str = "/api/v1/advisory/decision-jobs",
) -> dict[str, Any]:
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
            "status": f"{base_path}/{job['job_id']}",
            "result": f"{base_path}/{job['job_id']}/result",
            "events": f"{base_path}/{job['job_id']}/events",
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
