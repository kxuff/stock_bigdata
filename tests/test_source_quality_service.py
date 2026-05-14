import json
from pathlib import Path

from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle
from app.services.source_quality_service import assess_source_quality


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def load_sample(name: str) -> dict:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def test_source_quality_scores_complete_fresh_sources_high() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("normal_tool_results.json"))

    assessment = assess_source_quality(request, bundle)

    assert assessment.source_quality.overall_quality_score >= 0.85
    assert assessment.source_quality.freshness_score == 1.0
    assert assessment.source_quality_cap == 0.9
    assert assessment.stale_data is False
    assert assessment.quality_warnings == []


def test_source_quality_caps_stale_data() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("stale_data_tool_results.json"))

    assessment = assess_source_quality(request, bundle)

    assert assessment.stale_data is True
    assert assessment.source_quality.freshness_score < 0.6
    assert assessment.source_quality_cap == 0.55
    assert "One or more tool results are stale." in assessment.quality_warnings


def test_source_quality_warns_for_missing_optional_context() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("missing_valuation_tool_results.json"))

    assessment = assess_source_quality(request, bundle)

    assert assessment.source_quality.completeness_score < 1.0
    assert "Optional or per-symbol tool context is incomplete." in assessment.quality_warnings
