from typing import Any

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.schemas.decision import ErrorResponse, PortfolioDecision, SingleSymbolDecision
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultValidationError
from app.services.decision_service import AdvisoryDecisionService, DecisionValidationError
from app.services.tool_result_provider import SampleToolResultProvider, ToolResultProvider


app = FastAPI(title="Orca Agent Advisory API", version="0.1.0")


def get_decision_service() -> AdvisoryDecisionService:
    return AdvisoryDecisionService()


def get_tool_result_provider() -> ToolResultProvider:
    return SampleToolResultProvider()


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


async def _safe_json(request: Request) -> Any:
    try:
        return await request.json()
    except ValueError:
        return None


def _request_id_from_body(body: Any) -> str:
    if isinstance(body, dict) and isinstance(body.get("request_id"), str):
        return body["request_id"]
    return "UNKNOWN"
