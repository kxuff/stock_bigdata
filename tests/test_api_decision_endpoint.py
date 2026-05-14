import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import AgentSettings
from app.main import app, get_decision_service, get_tool_result_provider
from app.services.decision_service import AdvisoryDecisionService
from app.services.tool_result_provider import SampleToolResultProvider


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def load_sample(name: str) -> dict:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def test_decision_endpoint_returns_normal_final_json(tmp_path: Path) -> None:
    app.dependency_overrides[get_decision_service] = lambda: AdvisoryDecisionService(
        settings=AgentSettings(advisory_output_dir=tmp_path)
    )
    app.dependency_overrides[get_tool_result_provider] = lambda: SampleToolResultProvider()
    client = TestClient(app)

    response = client.post("/api/v1/advisory/decision", json=load_sample("normal_request.json"))

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "req_20260513_001"
    assert payload["decision_mode"] == "single_symbol_advisory"
    assert payload["not_financial_advice"] is True
    assert payload["confidence_breakdown"]["risk_adjusted_confidence"] == payload["confidence"]
    assert payload["audit"]["input_request_hash"].startswith("sha256:")


def test_decision_endpoint_returns_missing_required_tool_error() -> None:
    app.dependency_overrides[get_decision_service] = lambda: AdvisoryDecisionService()
    app.dependency_overrides[get_tool_result_provider] = lambda: SampleToolResultProvider()
    client = TestClient(app)
    request_payload = load_sample("normal_request.json")
    request_payload["metadata"]["tool_results_sample"] = "unavailable_market_tool_results.json"

    response = client.post("/api/v1/advisory/decision", json=request_payload)

    app.dependency_overrides.clear()
    assert response.status_code == 400
    payload = response.json()
    assert payload["status"] == "ERROR"
    assert payload["error_code"] == "MISSING_REQUIRED_TOOL_RESULT"
    assert payload["missing_tool_results"] == ["market_features"]


def test_decision_endpoint_rejects_invalid_request_shape() -> None:
    client = TestClient(app)
    request_payload = load_sample("normal_request.json")
    request_payload["symbols"] = ["AAPL", "MSFT"]

    response = client.post("/api/v1/advisory/decision", json=request_payload)

    assert response.status_code == 422
    payload = response.json()
    assert payload["status"] == "ERROR"
    assert payload["error_code"] == "INVALID_REQUEST"
    assert payload["request_id"] == "req_20260513_001"
