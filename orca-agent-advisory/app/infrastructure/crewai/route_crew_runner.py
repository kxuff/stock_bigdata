import json
import re
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
        if route.route == AgentRoute.MARKET_BRIEF:
            payload = self._ground_market_brief_payload(payload, request)
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

    def _ground_market_brief_payload(self, payload: RouteAgentResponseOutput, request: AgentQueryRequest) -> RouteAgentResponseOutput:
        """Keep agent prose, but force structured market data to come from tool/provider."""
        limit = _market_brief_limit(request.message)
        leaders = json.loads(json.dumps(self.market_screen_provider.screen_latest(limit), default=str))
        payload.result_type = "market_brief"
        result = dict(payload.result or {})
        result["leaders"] = leaders
        payload.result = result
        payload.message = _ground_market_brief_message(payload.message, leaders)
        return payload

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


def _market_brief_limit(message: str) -> int:
    text = str(message).lower()
    word_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }
    match = re.search(r"\b(?:top\s*)?(\d{1,2})\s*(?:stocks?|names?|tickers?)\b", text)
    if match:
        return max(1, min(int(match.group(1)), 20))
    for word, value in word_numbers.items():
        if re.search(rf"\b(?:top\s*)?{word}\s*(?:stocks?|names?|tickers?)\b", text):
            return value
    return 5


def _ground_market_brief_message(message: str, leaders: list[dict[str, Any]]) -> str:
    if not leaders:
        return message
    if not _message_matches_leaders(message, leaders):
        message = _fallback_market_brief_message(leaders)
    latest = _latest_as_of(leaders)
    if latest:
        message = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:Z)?)?\b", latest, message)
    if "Not financial advice." not in message:
        message = message.rstrip().rstrip(".") + ". Not financial advice."
    return message


def _message_matches_leaders(message: str, leaders: list[dict[str, Any]]) -> bool:
    leader_symbols = {_row_symbol(row) for row in leaders[:5]}
    leader_symbols.discard("")
    if not leader_symbols:
        return True
    mentioned = set(re.findall(r"\b[A-Z]{2,5}\b", message))
    ignored = {"ORCA", "RSI", "RSI14", "RVOL", "RVOL20"}
    mentioned -= ignored
    return not mentioned or bool(mentioned & leader_symbols)


def _fallback_market_brief_message(leaders: list[dict[str, Any]]) -> str:
    top = leaders[:5]
    names = ", ".join(_row_symbol(row) for row in top if _row_symbol(row))
    lead = top[0]
    lead_symbol = _row_symbol(lead)
    lead_score = _row_number(lead, "final_score")
    lead_price = _row_number(lead, "latest_price", "price", "entry_price")
    message = f"Top stocks to watch now: {names}."
    if lead_symbol and lead_score is not None:
        message += f" {lead_symbol} leads with ORCA score {lead_score:.2f}"
        if lead_price is not None:
            message += f" at {lead_price:.2f}"
        message += "."
    risk_notes = []
    for row in leaders:
        symbol = _row_symbol(row)
        rsi = _row_number(row, "RSI14")
        risk = _row_number(row, "risk_prob")
        if symbol and rsi is not None and rsi >= 70:
            risk_notes.append(f"{symbol} RSI14 {rsi:.1f}")
        elif symbol and risk is not None and risk >= 0.30:
            risk_notes.append(f"{symbol} risk_prob {risk:.2f}")
    if risk_notes:
        message += " Risk flags: " + "; ".join(risk_notes[:2]) + "."
    latest = _latest_as_of(leaders)
    if latest:
        message += f" Latest data as_of {latest}."
    return message


def _row_symbol(row: dict[str, Any]) -> str:
    return str(row.get("symbol") or row.get("Symbol") or "").upper()


def _row_number(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _latest_as_of(leaders: list[dict[str, Any]]) -> str | None:
    values = [str(row.get("as_of") or row.get("Datetime") or "") for row in leaders]
    values = [value for value in values if value]
    return max(values) if values else None
