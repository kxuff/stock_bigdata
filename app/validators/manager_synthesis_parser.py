from typing import Any

from pydantic import ValidationError

from app.schemas.manager_outputs import ManagerSynthesisOutput
from app.schemas.request import AdvisoryDecisionRequest
from app.validators.output_repair import parse_json_object, parse_model_output


class ManagerSynthesisParseError(ValueError):
    """Raised when manager output cannot be parsed into the required contract."""


def parse_manager_synthesis_output(
    raw_result: Any,
    request: AdvisoryDecisionRequest,
) -> ManagerSynthesisOutput:
    try:
        return parse_model_output(raw_result, ManagerSynthesisOutput)
    except ValidationError as exc:
        payload = parse_json_object(raw_result)
        normalized = normalize_manager_synthesis_payload(payload, request)
        try:
            return ManagerSynthesisOutput.model_validate(normalized)
        except ValidationError as normalized_exc:
            raise ManagerSynthesisParseError(str(normalized_exc)) from exc


def normalize_manager_synthesis_payload(
    payload: dict[str, Any],
    request: AdvisoryDecisionRequest,
) -> dict[str, Any]:
    normalized = {
        key: value
        for key, value in payload.items()
        if key in ManagerSynthesisOutput.model_fields
    }
    if "time_horizon" in normalized:
        normalized["time_horizon"] = _normalize_time_horizon(
            normalized["time_horizon"],
            request.user_context.investment_horizon.value,
        )
    normalized["supporting_signals"] = _string_list(
        normalized.get("supporting_signals", payload.get("supporting_signals"))
    )
    normalized["conflicting_signals"] = _string_list(
        normalized.get("conflicting_signals", payload.get("conflicting_signals"))
    )
    normalized["risk_warnings"] = _string_list(
        normalized.get("risk_warnings", payload.get("risk_warnings"))
    )
    normalized["limitations"] = _string_list(
        normalized.get("limitations", payload.get("limitations"))
    )
    normalized["data_citations"] = _string_list(
        normalized.get("data_citations", payload.get("data_citations"))
    )
    return normalized


def _normalize_time_horizon(value: Any, default: str) -> str:
    text = str(value or "").upper()
    if "SHORT" in text:
        return "SHORT_TERM"
    if "MEDIUM" in text or "MONTH" in text:
        return "MEDIUM_TERM"
    if "LONG" in text:
        return "LONG_TERM"
    if "DAY" in text or "INTRADAY" in text:
        return "INTRADAY"
    return default


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_stringify_item(item) for item in value if item]
    return [_stringify_item(value)]


def _stringify_item(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("source_ref", "source", "signal", "risk", "limitation", "description"):
            value = item.get(key)
            if value:
                return str(value)
    return str(item)
