import json
from pathlib import Path

from app.schemas.request import AdvisoryDecisionRequest
from app.validators.financial_validator import validate_financial_output


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def load_sample(name: str) -> dict:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def test_financial_validator_accepts_valid_portfolio_decision() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("portfolio_allocation_request.json"))
    payload = load_sample("portfolio_allocation_final_decision.json")

    result = validate_financial_output(payload, request=request)

    assert result.passed is True
    assert result.violations == []


def test_financial_validator_rejects_negative_allocation() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("portfolio_allocation_request.json"))
    payload = load_sample("portfolio_allocation_final_decision.json")
    payload["portfolio_allocation"][0]["weight_pct"] = -1

    result = validate_financial_output(payload, request=request)

    assert result.passed is False
    assert any("cannot be negative" in violation for violation in result.violations)


def test_financial_validator_rejects_allocation_total_not_100() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("portfolio_allocation_request.json"))
    payload = load_sample("portfolio_allocation_final_decision.json")
    payload["portfolio_allocation"][0]["weight_pct"] = 20

    result = validate_financial_output(payload, request=request)

    assert result.passed is False
    assert any("must total 100" in violation for violation in result.violations)


def test_financial_validator_rejects_max_single_asset_weight_violation() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("portfolio_allocation_request.json"))
    payload = load_sample("portfolio_allocation_final_decision.json")
    payload["portfolio_allocation"][0]["weight_pct"] = 45
    payload["portfolio_allocation"][2]["weight_pct"] = 20

    result = validate_financial_output(payload, request=request)

    assert result.passed is False
    assert any("max_single_asset_weight" in violation for violation in result.violations)


def test_financial_validator_rejects_missing_advice_boundary_and_guarantee_claim() -> None:
    payload = load_sample("normal_final_decision.json")
    payload["not_financial_advice"] = False
    payload["summary"] = "This has guaranteed profit."

    result = validate_financial_output(payload)

    assert result.passed is False
    assert "not_financial_advice must be true" in result.violations
    assert "forbidden financial claim: guaranteed profit" in result.violations
