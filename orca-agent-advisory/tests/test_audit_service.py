import json
from datetime import UTC, datetime
from pathlib import Path

from app.config import AgentSettings
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle
from app.application.services.audit_service import create_audit_metadata, create_retrieved_tool_audit


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def load_sample(name: str) -> dict:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def test_audit_metadata_contains_deterministic_hashes_and_run_context() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("normal_tool_results.json"))
    settings = AgentSettings(llm_model="openai/oc/deepseek-v4-flash-free", agent_temperature=0.2)
    created_at = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)

    audit = create_audit_metadata(
        request,
        bundle,
        settings=settings,
        run_id="run_test",
        validator_version="v1.0.0",
        created_at=created_at,
    )
    same_audit = create_audit_metadata(
        request,
        bundle,
        settings=settings,
        run_id="run_test_2",
        validator_version="v1.0.0",
        created_at=created_at,
    )

    assert audit.run_id == "run_test"
    assert audit.request_id == request.request_id
    assert audit.input_request_hash.startswith("sha256:")
    assert audit.tool_result_bundle_hash.startswith("sha256:")
    assert audit.input_request_hash == same_audit.input_request_hash
    assert audit.tool_result_bundle_hash == same_audit.tool_result_bundle_hash
    assert audit.validator_version == "v1.0.0"
    assert audit.created_at == created_at


def test_retrieved_tool_audit_hashes_each_available_tool_result() -> None:
    bundle = ToolResultBundle.model_validate(load_sample("normal_tool_results.json"))

    audit = create_retrieved_tool_audit(bundle)

    assert audit.tool_result_bundle_hash is not None
    assert audit.tool_result_bundle_hash.startswith("sha256:")
    assert {tool_call.tool for tool_call in audit.tool_calls} == {
        "MarketFeatureTool",
        "MlPredictionTool",
        "NewsSentimentTool",
        "FundamentalsTool",
        "RiskFeatureTool",
    }
    assert all(tool_call.result_hash.startswith("sha256:") for tool_call in audit.tool_calls)
