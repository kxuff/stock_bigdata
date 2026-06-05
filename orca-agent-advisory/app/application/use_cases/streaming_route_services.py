from dataclasses import dataclass
from typing import Any

from app.application.ports.streaming_alert_provider import StreamingAlertProvider
from app.application.ports.streaming_observability_provider import StreamingObservabilityProvider
from app.application.ports.streaming_quality_provider import StreamingQualityProvider
from app.schemas.agent import AgentQueryResponse, RoutedAgentQuery
from app.schemas.route_results import (
    StreamingAlertReviewResult,
    StreamingAlertRow,
    StreamingFeatureDriftResult,
    StreamingFeatureDriftRow,
    StreamingFreshnessResult,
    StreamingFreshnessRow,
    StreamingIngestionLagResult,
    StreamingIngestionLagRow,
    StreamingPipelineHealthResult,
    StreamingPipelineStage,
    StreamingQualityIncident,
    StreamingQualityIncidentsResult,
    StreamingSymbolMonitorResult,
    StreamingTopicInspectionResult,
    StreamingTopicSample,
)


@dataclass
class StreamingRouteServices:
    observability_provider: StreamingObservabilityProvider
    alert_provider: StreamingAlertProvider
    quality_provider: StreamingQualityProvider
    topic_inspection_provider: Any | None = None

    def pipeline_health(self, route: RoutedAgentQuery) -> AgentQueryResponse:
        result = StreamingPipelineHealthResult(stages=[StreamingPipelineStage(**_pick(row, {"stage", "table", "status", "latest_timestamp", "row_count", "error"})) for row in self.observability_provider.get_pipeline_health(60)])
        return _response(route, "streaming_pipeline_health", result.model_dump())

    def freshness_check(self, route: RoutedAgentQuery) -> AgentQueryResponse:
        result = StreamingFreshnessResult(rows=[StreamingFreshnessRow(**_pick(row, {"symbol", "table", "latest_timestamp", "lag_minutes", "status", "error"})) for row in self.observability_provider.get_symbol_freshness(route.symbols, 60)])
        return _response(route, "streaming_freshness_check", result.model_dump())

    def alert_review(self, route: RoutedAgentQuery) -> AgentQueryResponse:
        result = StreamingAlertReviewResult(alerts=[_alert(row) for row in self.alert_provider.get_latest_alerts(route.symbols, [], 50, 240)])
        return _response(route, "streaming_alert_review", result.model_dump())

    def symbol_monitor(self, route: RoutedAgentQuery) -> AgentQueryResponse:
        symbol = route.symbols[0] if route.symbols else None
        freshness = self.observability_provider.get_symbol_freshness([symbol] if symbol else [], 60)
        alerts = self.alert_provider.get_active_symbol_alerts(symbol, 240) if symbol else []
        result = StreamingSymbolMonitorResult(symbol=symbol, freshness=[StreamingFreshnessRow(**_pick(row, {"symbol", "table", "latest_timestamp", "lag_minutes", "status", "error"})) for row in freshness], alerts=[_alert(row) for row in alerts])
        return _response(route, "streaming_symbol_monitor", result.model_dump())

    def feature_drift(self, route: RoutedAgentQuery) -> AgentQueryResponse:
        result = StreamingFeatureDriftResult(rows=[StreamingFeatureDriftRow(**_pick(row, {"symbol", "feature", "streaming_value", "batch_value", "delta", "status", "error"})) for row in self.quality_provider.compare_streaming_to_batch_features(route.symbols, None)])
        return _response(route, "streaming_feature_drift", result.model_dump())

    def ingestion_lag(self, route: RoutedAgentQuery) -> AgentQueryResponse:
        result = StreamingIngestionLagResult(rows=[StreamingIngestionLagRow(**_pick(row, {"table", "latest_timestamp", "lag_minutes", "status", "error"})) for row in self.observability_provider.get_ingestion_lag(60)])
        return _response(route, "streaming_ingestion_lag", result.model_dump())

    def topic_inspection(self, route: RoutedAgentQuery) -> AgentQueryResponse:
        provider = self.topic_inspection_provider or self.observability_provider
        try:
            samples = provider.inspect_topics() if hasattr(provider, "inspect_topics") else []
        except Exception as exc:  # noqa: BLE001
            samples = [{"topic": "kafka", "status": "error", "error": str(exc), "limitation": "Kafka direct topic inspection failed soft."}]
        result = StreamingTopicInspectionResult(samples=[StreamingTopicSample(**_pick(row, {"topic", "status", "partition_count", "latest_offsets", "consumer_lag", "sample", "limitation", "error"})) for row in samples])
        return _response(route, "streaming_topic_inspection", result.model_dump())

    def quality_incidents(self, route: RoutedAgentQuery) -> AgentQueryResponse:
        result = StreamingQualityIncidentsResult(incidents=[StreamingQualityIncident(**_pick(row, {"symbol", "table", "incident_type", "message", "timestamp"})) for row in self.quality_provider.find_quality_incidents(route.symbols, 240, 50)])
        return _response(route, "streaming_quality_incidents", result.model_dump())


def _response(route: RoutedAgentQuery, result_type: str, result: dict[str, Any]) -> AgentQueryResponse:
    return AgentQueryResponse(route=route.route, status="immediate", message=route.message, symbols=route.symbols, result_type=result_type, result=result, suggested_actions=route.suggested_actions, router_confidence=route.confidence)


def _alert(row: dict[str, Any]) -> StreamingAlertRow:
    return StreamingAlertRow(symbol=_str(row.get("Symbol") or row.get("symbol")), severity=_str(row.get("severity") or row.get("Severity") or row.get("status")), alert_type=_str(row.get("alert_type") or row.get("type")), message=_str(row.get("message") or row.get("error")), timestamp=_str(row.get("Datetime") or row.get("timestamp") or row.get("event_time")))


def _pick(row: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: row.get(key) for key in keys if key in row}


def _str(value: Any) -> str | None:
    return None if value is None else str(value)
