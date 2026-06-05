from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from app.application.ports.query_router import QueryRouter
from app.application.ports.tool_result_provider import ToolResultProvider
from app.application.use_cases.advisory_decision_service import AdvisoryDecisionService
from app.application.use_cases.route_services import AgentRouteServices
from app.application.use_cases.streaming_route_services import StreamingRouteServices
from app.schemas.agent import AgentQueryRequest, AgentQueryResponse
from app.schemas.enums import AgentRoute, DecisionMode, InvestmentHorizon, RiskTolerance
from app.schemas.request import AdvisoryDecisionRequest, UserContext


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
            advisory_request = AdvisoryDecisionRequest(
                request_id=f"agent-{uuid4()}",
                timestamp=now,
                as_of_timestamp=now,
                user_query=request.message,
                decision_mode=DecisionMode.SINGLE_SYMBOL_ADVISORY,
                symbols=[routed.symbols[0]],
                user_context=UserContext(
                    risk_tolerance=RiskTolerance(request.context.risk_tolerance),
                    investment_horizon=InvestmentHorizon(request.context.investment_horizon),
                ),
                metadata={"source": "agent_query", "route": routed.route.value},
            )
            tool_results = self.tool_result_provider.get_tool_results(advisory_request)
            decision = self.advisory_service.decide(advisory_request, tool_results)
            return AgentQueryResponse(route=routed.route, status="immediate", message=routed.message, symbols=routed.symbols, result_type="single_symbol_decision", result=decision.model_dump(mode="json"), suggested_actions=routed.suggested_actions, router_confidence=routed.confidence)
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
        raise RuntimeError(f"unsupported agent route: {routed.route.value}")
