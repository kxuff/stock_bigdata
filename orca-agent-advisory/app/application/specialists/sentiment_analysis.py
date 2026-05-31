from app.schemas.agent_outputs import SentimentAgentOutput
from app.schemas.enums import AgentStatus, SentimentLabel, ToolStatus
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle


def analyze_sentiment(
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
) -> SentimentAgentOutput:
    sentiment_result = tool_results.sentiment_snapshot
    if sentiment_result is None or sentiment_result.status != ToolStatus.SUCCESS:
        return SentimentAgentOutput(
            status=AgentStatus.SKIPPED,
            summary="Sentiment context is unavailable; no sentiment inference was made.",
            confidence=0.0,
            missing_fields=["sentiment_snapshot"],
            limitations=["SENTIMENT_CONTEXT_UNAVAILABLE"],
            source_refs=[],
            sentiment_label=SentimentLabel.UNAVAILABLE,
            top_drivers=[],
        )

    source_refs = sentiment_result.source_refs
    missing_fields: list[str] = []
    top_drivers: list[str] = []
    labels: list[SentimentLabel] = []
    scores: list[float] = []
    for symbol in request.symbols:
        snapshot = sentiment_result.data.get(symbol)
        if snapshot is None:
            missing_fields.append(f"sentiment_snapshot.{symbol}")
            continue
        labels.append(snapshot.sentiment_label)
        scores.append(snapshot.sentiment_score)
        top_drivers.extend(snapshot.top_drivers)

    if not labels:
        return SentimentAgentOutput(
            status=AgentStatus.SKIPPED,
            summary="Sentiment snapshots did not include requested symbols.",
            confidence=0.0,
            missing_fields=missing_fields,
            limitations=["SENTIMENT_SYMBOL_CONTEXT_UNAVAILABLE"],
            source_refs=source_refs,
            sentiment_label=SentimentLabel.UNAVAILABLE,
            top_drivers=[],
        )

    average_abs_score = sum(abs(score) for score in scores) / len(scores)
    confidence = round(0.5 + min(average_abs_score, 1.0) * 0.3, 2)
    limitations = []
    if sentiment_result.freshness.is_stale:
        limitations.append("SENTIMENT_FRESHNESS_STALE")
        confidence = min(confidence, 0.45)
    return SentimentAgentOutput(
        status=AgentStatus.DEGRADED if missing_fields or limitations else AgentStatus.SUCCESS,
        summary="Sentiment was summarized from NewsSentimentTool snapshots.",
        confidence=confidence,
        missing_fields=missing_fields,
        limitations=limitations,
        source_refs=source_refs,
        sentiment_label=_dominant_sentiment_label(labels),
        top_drivers=list(dict.fromkeys(top_drivers)),
    )


def _dominant_sentiment_label(labels: list[SentimentLabel]) -> SentimentLabel:
    if len(set(labels)) > 1:
        return SentimentLabel.MIXED
    return labels[0]
