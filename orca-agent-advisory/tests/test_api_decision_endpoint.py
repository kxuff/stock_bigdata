import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import AgentSettings
from app.application.use_cases.advisory_decision_service import AdvisoryDecisionService
from app.main import app, get_decision_service, get_tool_result_provider
from app.infrastructure.bigdata.bigdata_ml_provider import BigdataMlToolResultProvider
from app.infrastructure.storage.output_store import DecisionOutputStore
from conftest import FixtureCrewRunner


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def load_sample(name: str) -> dict:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def bigdata_row(symbol: str = "AAPL") -> dict:
    return {
        "Symbol": symbol,
        "Datetime": "2026-05-13 15:00:00",
        "model_version": "xgb_v1",
        "pred_a": 0.541499137878418,
        "risk_prob": 0.0377756766974926,
        "final_score": 0.51,
        "feature_version": "price_v1_notebook_ac",
        "prediction_process_date": "2026-05-13 15:00:00",
        "source_feature_process_date": "2026-05-13 14:59:00",
        "Close": 442.25,
        "r1": 0.0125,
        "RVOL20": 1.34,
        "RSI14": 58.2,
        "MACD_hist": 0.17,
        "BB_pctB": 0.84,
        "BB_width": 0.11,
        "EMA20_50_spread": 2.1,
        "EMA20_slope": 0.4,
        "ROC10": 1.9,
        "ADX14": 24.0,
    }


def test_decision_endpoint_returns_normal_final_json(tmp_path: Path) -> None:
    app.dependency_overrides[get_decision_service] = lambda: AdvisoryDecisionService(
        settings=AgentSettings(advisory_output_dir=tmp_path),
        crew_runner=FixtureCrewRunner(),
        output_store=DecisionOutputStore(tmp_path),
    )
    app.dependency_overrides[get_tool_result_provider] = lambda: BigdataMlToolResultProvider(
        row_loader=lambda _: [bigdata_row()]
    )
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
    app.dependency_overrides[get_tool_result_provider] = lambda: BigdataMlToolResultProvider(row_loader=lambda _: [])
    client = TestClient(app)
    request_payload = load_sample("normal_request.json")

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


def test_decision_endpoint_runs_with_bigdata_ml_provider(tmp_path: Path) -> None:
    row = bigdata_row("ADBE") | {
        "Datetime": "2026-05-27 00:00:00",
        "prediction_process_date": "2026-05-27 23:12:00",
        "source_feature_process_date": "2026-05-27 22:59:00",
    }
    provider = BigdataMlToolResultProvider(row_loader=lambda _: [row])
    app.dependency_overrides[get_decision_service] = lambda: AdvisoryDecisionService(
        settings=AgentSettings(advisory_output_dir=tmp_path),
        crew_runner=FixtureCrewRunner(),
        output_store=DecisionOutputStore(tmp_path),
    )
    app.dependency_overrides[get_tool_result_provider] = lambda: provider
    client = TestClient(app)
    request_payload = load_sample("normal_request.json")
    request_payload.update(
        {
            "request_id": "req_bigdata_ml_api",
            "user_query": "Should I buy ADBE today?",
            "symbols": ["ADBE"],
        }
    )

    response = client.post("/api/v1/advisory/decision", json=request_payload)

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["decision_mode"] == "single_symbol_advisory"
    assert payload["symbol"] == "ADBE"
    assert any("ml_ready.stock_predictions" in ref for ref in payload["data_citations"])
    assert payload["not_financial_advice"] is True
