import json
from pathlib import Path

import pytest

from app.schemas.request import AdvisoryDecisionRequest
from app.validators.manager_synthesis_parser import (
    ManagerSynthesisParseError,
    parse_manager_synthesis_output,
)


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def load_sample(name: str) -> dict:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def test_manager_parser_normalizes_time_horizon_and_object_lists() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    raw = json.dumps(
        {
            "summary": "AAPL remains constructive but needs risk caps.",
            "time_horizon": "short-to-medium-term",
            "proposed_recommendation": "BUY",
            "decision_rationale": [],
            "supporting_signals": [{"signal": "technical trend is positive"}],
            "conflicting_signals": [{"signal": "valuation upside is limited"}],
            "risk_warnings": [{"risk": "sector concentration"}],
            "limitations": [{"limitation": "portfolio context unavailable"}],
            "data_citations": [{"source": "postgresql.real_time_prices:AAPL"}],
        }
    )

    output = parse_manager_synthesis_output(raw, request)

    assert output.time_horizon == "SHORT_TERM"
    assert output.proposed_recommendation == "BUY"
    assert output.supporting_signals == ["technical trend is positive"]
    assert output.risk_warnings == ["sector concentration"]
    assert output.data_citations == ["postgresql.real_time_prices:AAPL"]


def test_manager_parser_fails_when_recommendation_is_missing() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    raw = json.dumps(
        {
            "summary": "No actionable draft.",
            "time_horizon": "SHORT_TERM",
            "supporting_signals": [],
            "conflicting_signals": [],
            "risk_warnings": [],
            "limitations": [],
            "data_citations": [],
        }
    )

    with pytest.raises(ManagerSynthesisParseError):
        parse_manager_synthesis_output(raw, request)
