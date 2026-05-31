from app.schemas.agent_outputs import ValuationAgentOutput
from app.schemas.enums import AgentStatus, ToolStatus, ValuationLabel
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle


def analyze_valuation(
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
) -> ValuationAgentOutput:
    valuation_result = tool_results.valuation_snapshot
    if valuation_result is None or valuation_result.status != ToolStatus.SUCCESS:
        return ValuationAgentOutput(
            status=AgentStatus.SKIPPED,
            summary="Valuation context is unavailable; no valuation inference was made.",
            confidence=0.0,
            missing_fields=["valuation_snapshot"],
            limitations=["VALUATION_CONTEXT_UNAVAILABLE"],
            source_refs=[],
            valuation_label=ValuationLabel.UNKNOWN,
            valuation_drivers=[],
        )

    source_refs = valuation_result.source_refs
    missing_fields: list[str] = []
    labels: list[ValuationLabel] = []
    drivers: list[str] = []
    quality_limits: list[str] = []
    for symbol in request.symbols:
        snapshot = valuation_result.data.get(symbol)
        if snapshot is None:
            missing_fields.append(f"valuation_snapshot.{symbol}")
            continue
        labels.append(snapshot.valuation_label)
        if snapshot.pe_ratio is not None:
            drivers.append(f"{symbol} pe_ratio={snapshot.pe_ratio}")
        if snapshot.sector_pe_ratio is not None:
            drivers.append(f"{symbol} sector_pe_ratio={snapshot.sector_pe_ratio}")
        if snapshot.upside_downside_pct is not None:
            drivers.append(f"{symbol} upside_downside_pct={snapshot.upside_downside_pct}")
        if snapshot.valuation_method:
            drivers.append(f"{symbol} valuation_method={snapshot.valuation_method}")
        if snapshot.valuation_quality:
            quality = snapshot.valuation_quality.upper()
            drivers.append(f"{symbol} valuation_quality={snapshot.valuation_quality}")
            if quality == "LOW":
                quality_limits.append("VALUATION_QUALITY_LOW")
            elif quality == "UNKNOWN":
                quality_limits.append("VALUATION_QUALITY_UNKNOWN")
        if snapshot.sector_sample_count is not None:
            drivers.append(f"{symbol} sector_sample_count={snapshot.sector_sample_count}")

    if not labels:
        return ValuationAgentOutput(
            status=AgentStatus.SKIPPED,
            summary="Valuation snapshots did not include requested symbols.",
            confidence=0.0,
            missing_fields=missing_fields,
            limitations=["VALUATION_SYMBOL_CONTEXT_UNAVAILABLE"],
            source_refs=source_refs,
            valuation_label=ValuationLabel.UNKNOWN,
            valuation_drivers=[],
        )

    limitations = list(dict.fromkeys(quality_limits))
    if valuation_result.freshness.is_stale:
        limitations.append("VALUATION_FRESHNESS_STALE")
    label = _dominant_label(labels)
    confidence = 0.58
    if ValuationLabel.UNKNOWN in labels or "VALUATION_QUALITY_UNKNOWN" in limitations or label == ValuationLabel.UNKNOWN:
        confidence = 0.35
    elif "VALUATION_QUALITY_LOW" in limitations:
        confidence = 0.45
    if valuation_result.freshness.is_stale:
        confidence = min(confidence, 0.45)

    return ValuationAgentOutput(
        status=AgentStatus.DEGRADED if missing_fields or limitations else AgentStatus.SUCCESS,
        summary="Valuation was assessed only from FundamentalsTool snapshots.",
        confidence=confidence,
        missing_fields=missing_fields,
        limitations=limitations,
        source_refs=source_refs,
        valuation_label=label,
        valuation_drivers=drivers,
    )


def _dominant_valuation_label(labels: list[ValuationLabel]) -> ValuationLabel:
    if ValuationLabel.OVERVALUED in labels:
        return ValuationLabel.OVERVALUED
    if ValuationLabel.UNDERVALUED in labels:
        return ValuationLabel.UNDERVALUED
    if ValuationLabel.FAIRLY_VALUED in labels:
        return ValuationLabel.FAIRLY_VALUED
    return ValuationLabel.UNKNOWN


def _dominant_label(labels: list[ValuationLabel]) -> ValuationLabel:
    return _dominant_valuation_label(labels)
