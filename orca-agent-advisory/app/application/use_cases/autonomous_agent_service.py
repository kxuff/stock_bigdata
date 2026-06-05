from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.application.ports.query_router import QueryRouter
from app.application.ports.tool_result_provider import ToolResultProvider
from app.application.use_cases.advisory_decision_service import AdvisoryDecisionService
from app.application.use_cases.route_services import AgentRouteServices
from app.application.use_cases.streaming_route_services import StreamingRouteServices
from app.schemas.agent import AgentQueryRequest, AgentQueryResponse
from app.schemas.enums import AgentRoute
from app.schemas.request import AdvisoryDecisionRequest


@dataclass
class AutonomousAgentService:
    router: QueryRouter
    route_services: AgentRouteServices
    advisory_service: AdvisoryDecisionService
    tool_result_provider: ToolResultProvider
    streaming_route_services: StreamingRouteServices | None = None

    def query(self, request: AgentQueryRequest) -> AgentQueryResponse:
        routed = self.router.route(request)
        if routed.route in {AgentRoute.CLARIFICATION, AgentRoute.OUT_OF_SCOPE}:
            return AgentQueryResponse(route=routed.route, status="immediate", message=routed.message, symbols=routed.symbols, suggested_actions=routed.suggested_actions, router_confidence=routed.confidence)
        if routed.route == AgentRoute.SINGLE_SYMBOL_ADVISORY:
            now = datetime.now(UTC)
            advisory_request = AdvisoryDecisionRequest(request_id=f"agent-{uuid4()}", timestamp=now, as_of_timestamp=now, user_query=request.message, decision_mode="single_symbol_advisory", symbols=[routed.symbols[0]], user_context={"risk_tolerance": request.context.risk_tolerance, "investment_horizon": request.context.investment_horizon}, metadata={"source": "agent_query", "route": routed.route.value})
            tool_results = self.tool_result_provider.get_tool_results(advisory_request)
            decision = self.advisory_service.decide(advisory_request, tool_results)
            return AgentQueryResponse(route=routed.route, status="immediate", message=routed.message, symbols=routed.symbols, result_type="single_symbol_decision", result=decision.model_dump(mode="json"), suggested_actions=routed.suggested_actions, router_confidence=routed.confidence)
        if routed.route == AgentRoute.PORTFOLIO_RECOMMENDATION:
            now = datetime.now(UTC)
            symbols = _portfolio_symbols(request, routed)
            advisory_request = AdvisoryDecisionRequest(
                request_id=f"agent-{uuid4()}",
                timestamp=now,
                as_of_timestamp=now,
                user_query=request.message,
                decision_mode="portfolio_recommendation",
                symbols=symbols,
                user_context=_portfolio_user_context(request),
                metadata=_portfolio_metadata(request, routed),
            )
            tool_results = self.tool_result_provider.get_tool_results(advisory_request)
            decision = self.advisory_service.decide(advisory_request, tool_results)
            return AgentQueryResponse(route=routed.route, status="immediate", message=routed.message, symbols=symbols, result_type="portfolio_decision", result=decision.model_dump(mode="json"), suggested_actions=routed.suggested_actions, router_confidence=routed.confidence)
        if routed.route == AgentRoute.SYMBOL_COMPARISON:
            return self.route_services.symbol_comparison(routed)
        if routed.route == AgentRoute.WATCHLIST_REVIEW:
            return self.route_services.watchlist_review(routed)
        if routed.route == AgentRoute.UNIVERSE_SCREEN:
            return self.route_services.universe_screen(routed)
        if routed.route == AgentRoute.MARKET_BRIEF:
            return self.route_services.market_brief(routed)
        if routed.route == AgentRoute.DATA_DIAGNOSTICS:
            return self.route_services.data_diagnostics(routed)
        if routed.route == AgentRoute.PORTFOLIO_REBALANCE:
            return self.route_services.portfolio_rebalance(request, routed)
        if routed.route == AgentRoute.BACKTEST_ANALYSIS:
            return self.route_services.backtest_analysis(request, routed)
        if self.streaming_route_services is not None:
            if routed.route == AgentRoute.STREAMING_PIPELINE_HEALTH:
                return self.streaming_route_services.pipeline_health(routed)
            if routed.route == AgentRoute.STREAMING_FRESHNESS_CHECK:
                return self.streaming_route_services.freshness_check(routed)
            if routed.route == AgentRoute.STREAMING_ALERT_REVIEW:
                return self.streaming_route_services.alert_review(routed)
            if routed.route == AgentRoute.STREAMING_SYMBOL_MONITOR:
                return self.streaming_route_services.symbol_monitor(routed)
            if routed.route == AgentRoute.STREAMING_FEATURE_DRIFT:
                return self.streaming_route_services.feature_drift(routed)
            if routed.route == AgentRoute.STREAMING_INGESTION_LAG:
                return self.streaming_route_services.ingestion_lag(routed)
            if routed.route == AgentRoute.STREAMING_TOPIC_INSPECTION:
                return self.streaming_route_services.topic_inspection(routed)
            if routed.route == AgentRoute.STREAMING_QUALITY_INCIDENTS:
                return self.streaming_route_services.quality_incidents(routed)
        return AgentQueryResponse(route=routed.route, status="immediate", message="Route recognized, but backend workflow is not implemented yet.", symbols=routed.symbols, suggested_actions=routed.suggested_actions, router_confidence=routed.confidence)


def _portfolio_symbols(request: AgentQueryRequest, route) -> list[str]:
    symbols = [*route.symbols, *request.context.symbols]
    if request.context.symbol:
        symbols.append(request.context.symbol)
    symbols.extend(_holding_symbols(request.context.metadata))
    normalized: list[str] = []
    for value in symbols:
        symbol = str(value or "").strip().upper().replace(".", "-")
        if symbol and symbol != "CASH" and symbol not in normalized:
            normalized.append(symbol)
    return normalized


def _portfolio_user_context(request: AgentQueryRequest) -> dict[str, Any]:
    metadata = request.context.metadata
    custom_constraints: dict[str, Any] = {}
    min_cash_weight = _float_or_none(metadata.get("min_cash_weight"))
    if min_cash_weight is not None:
        custom_constraints["min_cash_weight"] = min_cash_weight
    return {
        "risk_tolerance": request.context.risk_tolerance,
        "investment_horizon": request.context.investment_horizon,
        "target_sectors": _string_list(metadata.get("target_sectors")),
        "excluded_symbols": _symbol_list(metadata.get("excluded_symbols")),
        "max_single_asset_weight": _float_or_none(metadata.get("max_single_asset_weight")) or 40.0,
        "allow_cash_position": _bool_value(metadata.get("allow_cash_position"), True),
        "custom_constraints": custom_constraints,
    }


def _portfolio_metadata(request: AgentQueryRequest, route) -> dict[str, Any]:
    metadata = dict(request.context.metadata)
    metadata.update({"source": "agent_query", "route": route.route.value})
    return metadata


def _holding_symbols(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("holdings") or metadata.get("portfolio")
    if isinstance(raw, dict):
        raw = raw.get("holdings") or raw.get("positions") or raw.get("assets")
    if not isinstance(raw, list):
        return []
    symbols: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or item.get("ticker") or "").strip().upper().replace(".", "-")
        if symbol and symbol != "CASH":
            symbols.append(symbol)
    return symbols


def _symbol_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().upper().replace(".", "-") for item in value if str(item).strip()]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
