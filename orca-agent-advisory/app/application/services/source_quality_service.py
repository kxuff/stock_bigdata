from pydantic import BaseModel, ConfigDict, Field

from app.schemas.decision import SourceQuality
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import BaseToolResult, ToolResultBundle


class SourceQualityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_quality: SourceQuality
    source_quality_cap: float = Field(ge=0.0, le=1.0)
    source_quality_adjustment: float = Field(ge=-1.0, le=1.0)
    stale_data: bool
    quality_warnings: list[str] = Field(default_factory=list)


def assess_source_quality(
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
) -> SourceQualityAssessment:
    available_results = _available_tool_results(tool_results)
    required_results = [tool_results.market_features]

    stale_data = any(result.freshness.is_stale for result in available_results)
    freshness_score = _freshness_score(available_results)
    relevance_score = _relevance_score(request, available_results)
    completeness_score = _completeness_score(request, tool_results)
    source_reliability_score = _source_reliability_score(available_results)
    overall_quality_score = round(
        (
            0.35 * freshness_score
            + 0.25 * relevance_score
            + 0.25 * completeness_score
            + 0.15 * source_reliability_score
        ),
        2,
    )

    warnings: list[str] = []
    if any(result is None for result in required_results):
        warnings.append("Required market feature data is missing.")
    if stale_data:
        warnings.append("One or more tool results are stale.")
    if relevance_score < 0.7:
        warnings.append("Some source references do not directly mention requested symbols.")
    if completeness_score < 0.7:
        warnings.append("Optional or per-symbol tool context is incomplete.")

    source_quality = SourceQuality(
        overall_quality_score=overall_quality_score,
        freshness_score=freshness_score,
        relevance_score=relevance_score,
        completeness_score=completeness_score,
    )

    return SourceQualityAssessment(
        source_quality=source_quality,
        source_quality_cap=_source_quality_cap(overall_quality_score, freshness_score),
        source_quality_adjustment=round(min(0.0, (overall_quality_score - 1.0) * 0.2), 2),
        stale_data=stale_data,
        quality_warnings=warnings,
    )


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


def _freshness_score(results: list[BaseToolResult]) -> float:
    if not results:
        return 0.0
    fresh_count = sum(1 for result in results if not result.freshness.is_stale)
    return round(fresh_count / len(results), 2)


def _relevance_score(request: AdvisoryDecisionRequest, results: list[BaseToolResult]) -> float:
    source_refs = [source_ref for result in results for source_ref in result.source_refs]
    if not source_refs:
        return 0.5

    requested_symbols = set(request.symbols)
    relevant_refs = sum(
        1
        for source_ref in source_refs
        if any(symbol in source_ref.upper() for symbol in requested_symbols)
    )
    return round(relevant_refs / len(source_refs), 2)


def _completeness_score(
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
) -> float:
    checks: list[bool] = []

    market_data = tool_results.market_features.data if tool_results.market_features else {}
    checks.extend(symbol in market_data for symbol in request.symbols)

    optional_symbol_results = [
        tool_results.ml_predictions,
        tool_results.sentiment_snapshot,
        tool_results.valuation_snapshot,
        tool_results.risk_snapshot,
    ]
    for result in optional_symbol_results:
        data = result.data if result is not None else {}
        checks.extend(symbol in data for symbol in request.symbols)

    if request.decision_mode == "portfolio_recommendation":
        checks.append(tool_results.portfolio_snapshot is not None)

    if not checks:
        return 0.0
    return round(sum(1 for check in checks if check) / len(checks), 2)


def _source_reliability_score(results: list[BaseToolResult]) -> float:
    source_refs = [source_ref.lower() for result in results for source_ref in result.source_refs]
    if not source_refs:
        return 0.5

    trusted_prefixes = (
        "postgresql.",
        "spark.",
        "mongodb.",
        "nessie.",
        "raw.",
        "curated.",
        "ml_ready.",
        "bronze.",
        "silver.",
        "ml.",
        "alert.",
    )
    trusted_count = sum(
        1 for source_ref in source_refs if source_ref.startswith(trusted_prefixes)
    )
    return round(trusted_count / len(source_refs), 2)


def _source_quality_cap(overall_quality_score: float, freshness_score: float) -> float:
    if freshness_score < 0.6:
        return 0.55
    if overall_quality_score < 0.6:
        return 0.55
    if overall_quality_score < 0.7:
        return 0.65
    return 0.9
