from app.application.ports.crew_orchestrator import CrewOrchestrator
from app.application.ports.output_store import DecisionOutputStore as DecisionOutputStorePort
from app.application.ports.tool_result_provider import ToolResultProvider
from app.application.services.agent_query_router_service import AgentQueryRouterService
from app.application.use_cases.advisory_decision_service import AdvisoryDecisionService
from app.application.use_cases.autonomous_agent_service import AutonomousAgentService
from app.application.use_cases.route_services import AgentRouteServices
from app.application.use_cases.streaming_route_services import StreamingRouteServices
from app.config import AgentSettings, load_settings
from app.infrastructure.bigdata.bigdata_ml_provider import BigdataMlToolResultProvider
from app.infrastructure.bigdata.market_screen_provider import BigdataMarketScreenProvider
from app.infrastructure.bigdata.streaming_providers import SparkStreamingProvider
from app.infrastructure.crewai.crew_runner import HierarchicalCrewRunner
from app.infrastructure.llm.agent_route_planner import LiteLLMAgentRoutePlanner
from app.infrastructure.storage.output_store import DecisionOutputStore


def build_tool_result_provider(settings: AgentSettings) -> ToolResultProvider:
    return BigdataMlToolResultProvider()


def build_crew_orchestrator(settings: AgentSettings) -> CrewOrchestrator:
    return HierarchicalCrewRunner(settings=settings, verbose=settings.crewai_verbose)


def build_output_store(settings: AgentSettings) -> DecisionOutputStorePort:
    return DecisionOutputStore(settings.advisory_output_dir)


def build_decision_service(settings: AgentSettings | None = None) -> AdvisoryDecisionService:
    resolved_settings = settings or load_settings()
    return AdvisoryDecisionService(
        settings=resolved_settings,
        crew_runner=build_crew_orchestrator(resolved_settings),
        output_store=build_output_store(resolved_settings),
    )


def build_agent_route_planner(settings: AgentSettings | None = None) -> LiteLLMAgentRoutePlanner:
    return LiteLLMAgentRoutePlanner(settings=settings or load_settings())


def build_agent_query_router_service(settings: AgentSettings | None = None) -> AgentQueryRouterService:
    return AgentQueryRouterService(planner=build_agent_route_planner(settings or load_settings()))


def build_market_screen_provider(settings: AgentSettings | None = None) -> BigdataMarketScreenProvider:
    return BigdataMarketScreenProvider()


def build_route_services(settings: AgentSettings | None = None) -> AgentRouteServices:
    return AgentRouteServices(market_screen_provider=build_market_screen_provider(settings or load_settings()))


def build_streaming_provider(settings: AgentSettings | None = None) -> SparkStreamingProvider:
    return SparkStreamingProvider()


def build_streaming_route_services(settings: AgentSettings | None = None) -> StreamingRouteServices:
    provider = build_streaming_provider(settings or load_settings())
    return StreamingRouteServices(
        observability_provider=provider,
        alert_provider=provider,
        quality_provider=provider,
        topic_inspection_provider=provider,
    )


def build_autonomous_agent_service(settings: AgentSettings | None = None) -> AutonomousAgentService:
    resolved_settings = settings or load_settings()
    return AutonomousAgentService(
        router=build_agent_query_router_service(resolved_settings),
        route_services=build_route_services(resolved_settings),
        advisory_service=build_decision_service(resolved_settings),
        tool_result_provider=build_tool_result_provider(resolved_settings),
        streaming_route_services=build_streaming_route_services(resolved_settings),
    )
