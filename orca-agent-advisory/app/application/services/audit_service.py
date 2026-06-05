import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from app.config import AgentSettings, load_settings
from app.schemas.audit import AuditMetadata, RetrievedToolAudit, ToolCallAudit
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import BaseToolResult, ToolResultBundle


DEFAULT_VALIDATOR_VERSION = "v1.0.0"


def create_audit_metadata(
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
    *,
    settings: AgentSettings | None = None,
    run_id: str | None = None,
    validator_version: str = DEFAULT_VALIDATOR_VERSION,
    created_at: datetime | None = None,
) -> AuditMetadata:
    resolved_settings = settings or load_settings()
    return AuditMetadata(
        run_id=run_id or f"run_{uuid.uuid4().hex}",
        request_id=request.request_id,
        model_name=resolved_settings.llm_model,
        temperature=resolved_settings.agent_temperature,
        input_request_hash=hash_payload(request.model_dump(mode="json")),
        tool_result_bundle_hash=hash_payload(tool_results.model_dump(mode="json", exclude_none=True)),
        validator_version=validator_version,
        created_at=created_at or datetime.now(UTC),
    )


def create_retrieved_tool_audit(tool_results: ToolResultBundle) -> RetrievedToolAudit:
    results = _available_tool_results(tool_results)
    return RetrievedToolAudit(
        tool_calls=[
            ToolCallAudit(
                tool=result.tool,
                status=result.status,
                source_refs=result.source_refs,
                result_hash=hash_payload(result.model_dump(mode="json")),
            )
            for result in results
        ],
        tool_result_bundle_hash=hash_payload(tool_results.model_dump(mode="json", exclude_none=True)),
    )


def hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _available_tool_results(tool_results: ToolResultBundle) -> list[BaseToolResult]:
    return [
        result
        for result in [
            tool_results.market_features,
            tool_results.ml_predictions,
            tool_results.sentiment_snapshot,
            tool_results.valuation_snapshot,
            tool_results.risk_snapshot,
            tool_results.portfolio_snapshot,
        ]
        if result is not None
    ]
