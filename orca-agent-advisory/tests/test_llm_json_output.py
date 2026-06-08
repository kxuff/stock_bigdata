import json
import os
from pathlib import Path

import pytest

from app.config import load_settings
from app.infrastructure.llm.llm_factory import CrewAIUnavailableError, create_llm
from app.schemas.decision import SingleSymbolDecision
from app.validators.output_repair import JsonOutputParseError, parse_json_object, parse_model_output


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def load_sample(name: str) -> dict:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def test_parse_json_object_repairs_markdown_fence() -> None:
    assert parse_json_object('```json\n{"status": "ok"}\n```') == {"status": "ok"}


def test_parse_json_object_extracts_first_object_from_text() -> None:
    raw = 'Final answer:\n{"status": "ok", "confidence": 0.8}\nDone.'

    assert parse_json_object(raw) == {"status": "ok", "confidence": 0.8}


def test_parse_model_output_validates_pydantic_contract() -> None:
    sample = load_sample("normal_final_decision.json")
    raw = "```json\n" + json.dumps(sample) + "\n```"

    decision = parse_model_output(raw, SingleSymbolDecision)

    assert decision.request_id == "req_20260513_001"
    assert decision.not_financial_advice is True


def test_invalid_llm_json_raises_parse_error() -> None:
    with pytest.raises(JsonOutputParseError):
        parse_json_object("not json")


@pytest.mark.live
def test_live_9router_json_smoke_when_api_key_available() -> None:
    settings = load_settings()
    if settings.llm_api_key is None and not os.getenv("NINEROUTER_KEY"):
        pytest.skip("NINEROUTER_KEY is not configured")

    try:
        llm = create_llm(settings)
    except CrewAIUnavailableError as exc:
        pytest.skip(str(exc))

    response = llm.call(
        messages=[
            {
                "role": "user",
                "content": 'Return exactly this JSON object and nothing else: {"status":"ok"}',
            }
        ]
    )

    assert parse_json_object(response)["status"] == "ok"
