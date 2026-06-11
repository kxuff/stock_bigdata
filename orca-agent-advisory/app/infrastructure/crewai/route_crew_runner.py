import json
from dataclasses import dataclass, field
from typing import Any, Callable

from crewai import Agent as CrewAgent
from crewai import Crew, Process, Task as CrewTask
from crewai.tools import BaseTool

from app.application.ports.market_screen_provider import MarketScreenProvider
from app.application.ports.streaming_observability_provider import StreamingObservabilityProvider
from app.config import AgentSettings, load_settings
from app.infrastructure.crewai.config_loader import crewai_route_task_config, route_agent_config
from app.infrastructure.llm.llm_factory import create_llm
from app.schemas.agent import AgentQueryRequest, AgentQueryResponse, RoutedAgentQuery, SuggestedAction
from app.schemas.enums import AgentRoute
from app.schemas.route_agent_outputs import RouteAgentResponseOutput
from app.validators.output_repair import parse_model_output


@dataclass
class RouteCrewRunner:
    market_screen_provider: MarketScreenProvider
    streaming_observability_provider: StreamingObservabilityProvider | None = None
    settings: AgentSettings = field(default_factory=load_settings)
    llm_factory: Callable[[AgentSettings], Any] = create_llm
    verbose: bool = False

    def run(self, request: AgentQueryRequest, route: RoutedAgentQuery) -> AgentQueryResponse:
        llm = self.llm_factory(self.settings)
        tools = self._tools()
        agent = CrewAgent(
            config=route_agent_config("route_response_agent"),
            llm=llm,
            tools=tools,
            verbose=self.verbose,
            allow_delegation=False,
            max_execution_time=self.settings.agent_timeout_seconds,
        )
        task = CrewTask(
            config=crewai_route_task_config(_task_name(route.route)),
            agent=agent,
            output_pydantic=RouteAgentResponseOutput,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
            tracing=self.settings.crewai_tracing,
            share_crew=self.settings.crewai_share_crew,
        )
        raw = crew.kickoff(
            inputs={
                "user_query": request.message,
                "route": route.route.value,
                "symbols": ", ".join(route.symbols),
                "symbols_json": json.dumps(route.symbols),
                "risk_tolerance": request.context.risk_tolerance,
                "investment_horizon": request.context.investment_horizon,
            }
        )
        payload = _parse_payload(_extract_task_payload(task) or raw)
        return AgentQueryResponse(
            route=route.route,
            status="immediate",
            message=payload.message,
            symbols=route.symbols,
            result_type=payload.result_type,
            result=payload.result,
            suggested_actions=route.suggested_actions or [SuggestedAction(label="Ask for single-symbol advisory", route=AgentRoute.SINGLE_SYMBOL_ADVISORY)],
            router_confidence=route.confidence,
        )

    def _tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = [
            _MarketScreenTool(provider=self.market_screen_provider),
            _SymbolLoadTool(provider=self.market_screen_provider),
            _MarketDiagnosticsTool(provider=self.market_screen_provider),
        ]
        if self.streaming_observability_provider is not None:
            tools.append(_FreshnessTool(provider=self.streaming_observability_provider))
            tools.append(_PipelineHealthTool(provider=self.streaming_observability_provider))
        return tools


class _MarketScreenTool(BaseTool):
    name: str = "MarketScreenTool"
    description: str = "Read-only tool. Input: integer limit as text. Returns latest ranked ORCA market candidates with final_score, price, RSI14, RVOL20, risk_prob, and as_of."
    provider: Any

    def _run(self, query: str = "10") -> str:
        try:
            limit = int(str(query).strip() or "10")
        except ValueError:
            limit = 10
        return json.dumps(self.provider.screen_latest(max(1, min(limit, 50))), default=str)


class _SymbolLoadTool(BaseTool):
    name: str = "SymbolLoadTool"
    description: str = "Read-only tool. Input: comma-separated symbols. Returns ORCA market signal rows for requested symbols with source-backed metrics."
    provider: Any

    def _run(self, query: str = "") -> str:
        symbols = [s.strip().upper().replace(".", "-") for s in str(query).split(",") if s.strip()]
        return json.dumps(self.provider.load_symbols(symbols), default=str)


class _MarketDiagnosticsTool(BaseTool):
    name: str = "MarketDiagnosticsTool"
    description: str = "Read-only tool. Returns diagnostics for the market screening data source."
    provider: Any

    def _run(self, query: str = "") -> str:
        return json.dumps(self.provider.diagnose(), default=str)


class _FreshnessTool(BaseTool):
    name: str = "FreshnessTool"
    description: str = "Read-only tool. Input: comma-separated symbols. Returns per-symbol/table freshness status, latest timestamp, lag, and errors."
    provider: Any

    def _run(self, query: str = "") -> str:
        symbols = [s.strip().upper().replace(".", "-") for s in str(query).split(",") if s.strip()]
        return json.dumps(self.provider.get_symbol_freshness(symbols, 60), default=str)


class _PipelineHealthTool(BaseTool):
    name: str = "PipelineHealthTool"
    description: str = "Read-only tool. Returns recent streaming/batch pipeline health rows."
    provider: Any

    def _run(self, query: str = "") -> str:
        return json.dumps(self.provider.get_pipeline_health(60), default=str)


def _task_name(route: AgentRoute) -> str:
    return {
        AgentRoute.SYMBOL_COMPARISON: "route_symbol_comparison_task",
        AgentRoute.WATCHLIST_REVIEW: "route_watchlist_review_task",
        AgentRoute.UNIVERSE_SCREEN: "route_universe_screen_task",
        AgentRoute.MARKET_BRIEF: "route_market_brief_task",
        AgentRoute.DATA_DIAGNOSTICS: "route_data_diagnostics_task",
        AgentRoute.STREAMING_FRESHNESS_CHECK: "route_streaming_freshness_task",
    }.get(route, "route_market_brief_task")


def _extract_task_payload(task: Any | None) -> Any | None:
    if task is None:
        return None
    for attr in ("output", "result", "raw_output", "raw", "response"):
        value = getattr(task, attr, None)
        if value is not None:
            return value
    return None


def _parse_payload(payload: Any) -> RouteAgentResponseOutput:
    pydantic_output = getattr(payload, "pydantic", None)
    if isinstance(pydantic_output, RouteAgentResponseOutput):
        return pydantic_output
    if isinstance(payload, RouteAgentResponseOutput):
        return payload
    if isinstance(payload, dict):
        return RouteAgentResponseOutput.model_validate(payload)
    return parse_model_output(payload, RouteAgentResponseOutput)
