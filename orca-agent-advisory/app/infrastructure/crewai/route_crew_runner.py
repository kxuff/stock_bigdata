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
from app.schemas.route_results import PortfolioRebalanceResult
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
        tools = self._tools(route.route)
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
                "portfolio_json": json.dumps(_portfolio_context(request), default=str),
            }
        )
        payload = _parse_payload(_extract_task_payload(task) or raw)
        if route.route == AgentRoute.SYMBOL_COMPARISON:
            payload = self._ground_symbol_comparison_payload(payload, route.symbols)
        if route.route in {AgentRoute.MARKET_BRIEF, AgentRoute.TOP_STOCKS}:
            payload = self._ground_market_screen_payload(payload, request, route.route)
        if route.route == AgentRoute.PORTFOLIO_REBALANCE:
            payload = self._ground_portfolio_rebalance_payload(payload, request)
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
        return self._ground_market_screen_payload(payload, request, AgentRoute.MARKET_BRIEF)

    def _ground_symbol_comparison_payload(self, payload: RouteAgentResponseOutput, symbols: list[str], grounded_rows: list[dict[str, Any]] | None = None) -> RouteAgentResponseOutput:
        """Keep comparison rows source-backed; never trust agent-generated metrics/dates."""
        rows = grounded_rows or json.loads(json.dumps(self.market_screen_provider.load_symbols(symbols), default=str))
        ranked = sorted(rows, key=lambda row: row.get("final_score") or 0, reverse=True)
        comparison_rows: list[dict[str, Any]] = []
        for index, row in enumerate(ranked, start=1):
            warnings = _row_warnings(row)
            comparison_rows.append(
                {
                    "symbol": _row_symbol(row),
                    "rank": index,
                    "final_score": row.get("final_score"),
                    "latest_price": row.get("latest_price") or row.get("Close"),
                    "RSI14": row.get("RSI14"),
                    "RVOL20": row.get("RVOL20"),
                    "risk_prob": row.get("risk_prob"),
                    "as_of": row.get("as_of") or row.get("Datetime"),
                    "status": "warning" if warnings else "ok",
                    "warnings": warnings,
                }
            )
        payload.result_type = "symbol_comparison"
        payload.result = {"rows": comparison_rows, "source_refs": [f"symbol_load_tool:{','.join(symbols)}"]}
        if not _symbol_comparison_message_is_grounded(payload.message, comparison_rows):
            payload.message = _symbol_comparison_message(comparison_rows)
        return payload

    def _ground_market_screen_payload(
        self,
        payload: RouteAgentResponseOutput,
        request: AgentQueryRequest,
        route: AgentRoute,
        grounded_rows: list[dict[str, Any]] | None = None,
    ) -> RouteAgentResponseOutput:
        """Keep agent prose, but force structured market data to come from tool/provider."""
        default = 10
        limit = _market_screen_limit(request.message, default=default)
        rows = grounded_rows or json.loads(json.dumps(self.market_screen_provider.screen_latest(limit), default=str))
        payload.result_type = "market_brief" if route == AgentRoute.MARKET_BRIEF else "top_stocks"
        result = dict(payload.result or {})
        if route == AgentRoute.MARKET_BRIEF:
            result["leaders"] = rows
        else:
            result["stocks"] = rows
        payload.result = result
        try:
            _validate_agent_market_message(payload.message, rows)
        except ValueError:
            payload.message = self._repair_market_message(payload.message, request, route, result)
            _validate_agent_market_message(payload.message, rows, check_symbols=False)
        return payload

    def _repair_market_message(self, message: str, request: AgentQueryRequest, route: AgentRoute, result: dict[str, Any]) -> str:
        import litellm

        rows_key = "leaders" if route == AgentRoute.MARKET_BRIEF else "stocks"
        rows = result.get(rows_key) or []
        allowed_symbols = [_row_symbol(row) for row in rows if _row_symbol(row)]
        response = litellm.completion(
            model=self.settings.llm_model,
            api_key=self.settings.llm_api_key.get_secret_value() if self.settings.llm_api_key else None,
            api_base=self.settings.llm_base_url,
            temperature=self.settings.agent_temperature,
            timeout=min(self.settings.agent_timeout_seconds, 30),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the assistant market response using only the grounded JSON rows and optional news/sentiment fields. "
                        "Do not invent symbols, prices, news, dates, or metrics. Do not mention raw dates, as_of, Datetime, or freshness timestamps. "
                        "Mention only ticker symbols from the allowed_symbols list. "
                        "Write natural language for a human investor, not a quant dashboard. Avoid raw field names such as final_score, risk_prob, RSI14, RVOL20, pred_a, r1/r3/r5. "
                        "Convert 0..1 scores into /100 only when useful, and explain signals as plain meaning: low model risk, not overbought, weak volume confirmation, bullish/bearish news tone. "
                        "Do not add financial-advice disclaimers. Avoid vague labels like 'hot themes' or 'main read'; use plain labels like Market tone, Recent news, and What to watch. "
                        "Explain like the reader is not a finance professional. Avoid unexplained jargon such as capex, semis, infra, follow-through, overheating, tape, breadth, breakout, or crowded. If a term is necessary, explain it in the same sentence. "
                        "Every bullet or sentence must answer at least one of: what it means, which stock/group it affects, why it matters, or what to watch next. Do not output bare theme phrases like 'AI demand', 'cloud growth', 'chip demand', or 'earnings momentum' without explaining impact. "
                        "Use 3-4 clear sections for market briefs: Market tone, Leaders, Recent news, and What to watch. Each section should have 2-4 concise sentences. For ranked top-stocks lists, use 1-2 concise sentences per stock. Return plain text only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_query": request.message,
                            "route": route.value,
                            "invalid_message": message,
                            "allowed_symbols": allowed_symbols,
                            rows_key: rows,
                            "result": {k: v for k, v in result.items() if k != rows_key},
                        },
                        default=str,
                    ),
                },
            ],
        )
        return str(response["choices"][0]["message"]["content"]).strip()

    def _ground_portfolio_rebalance_payload(self, payload: RouteAgentResponseOutput, request: AgentQueryRequest) -> RouteAgentResponseOutput:
        payload = _validate_portfolio_rebalance_payload(payload, request, validate_message=False)
        if _invalid_portfolio_message(payload.message):
            payload.message = self._repair_portfolio_message(payload.message, request, payload.result)
            payload.result["message"] = payload.message
        return _validate_portfolio_rebalance_payload(payload, request, validate_message=True)

    def _repair_portfolio_message(self, message: str, request: AgentQueryRequest, result: dict[str, Any]) -> str:
        import litellm

        response = litellm.completion(
            model=self.settings.llm_model,
            api_key=self.settings.llm_api_key.get_secret_value() if self.settings.llm_api_key else None,
            api_base=self.settings.llm_base_url,
            temperature=self.settings.agent_temperature,
            timeout=min(self.settings.agent_timeout_seconds, 30),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the portfolio rebalance explanation using only the provided portfolio context and rebalance result. "
                        "Do not invent market metrics, dates, scores, RSI, RVOL, risk_prob, final_score, source_refs, or as_of timestamps. "
                        "Do not claim trades were executed. Explain in natural advisor-style paragraphs: concentration, proposed weight changes, cash buffer, and what to review before acting. Return plain text only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_query": request.message,
                            "invalid_message": message,
                            "portfolio_context": _portfolio_context(request),
                            "rebalance_result": result,
                        },
                        default=str,
                    ),
                },
            ],
        )
        return str(response["choices"][0]["message"]["content"]).strip()

    def _tools(self, route: AgentRoute) -> list[BaseTool]:
        tools: list[BaseTool] = [
            _MarketScreenTool(provider=self.market_screen_provider),
            _SymbolLoadTool(provider=self.market_screen_provider),
            _MarketDiagnosticsTool(provider=self.market_screen_provider),
        ]
        if self.streaming_observability_provider is not None and _is_streaming_route(route):
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
        AgentRoute.TOP_STOCKS: "route_top_stocks_task",
        AgentRoute.PORTFOLIO_REBALANCE: "route_portfolio_rebalance_task",
        AgentRoute.DATA_DIAGNOSTICS: "route_data_diagnostics_task",
        AgentRoute.STREAMING_FRESHNESS_CHECK: "route_streaming_freshness_task",
    }.get(route, "route_market_brief_task")


def _is_streaming_route(route: AgentRoute) -> bool:
    return route in {
        AgentRoute.STREAMING_PIPELINE_HEALTH,
        AgentRoute.STREAMING_FRESHNESS_CHECK,
        AgentRoute.STREAMING_ALERT_REVIEW,
        AgentRoute.STREAMING_SYMBOL_MONITOR,
        AgentRoute.STREAMING_FEATURE_DRIFT,
        AgentRoute.STREAMING_INGESTION_LAG,
        AgentRoute.STREAMING_TOPIC_INSPECTION,
        AgentRoute.STREAMING_QUALITY_INCIDENTS,
    }


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


def _portfolio_context(request: AgentQueryRequest) -> dict[str, Any]:
    metadata = request.context.metadata or {}
    holdings = metadata.get("portfolio") or metadata.get("holdings") or []
    return {
        "holdings": holdings,
        "constraints": {
            "max_single_asset_weight": metadata.get("max_single_asset_weight", 30),
            "min_cash_weight": metadata.get("min_cash_weight", 5),
            "excluded_symbols": metadata.get("excluded_symbols") or [],
            "allowed_symbols": metadata.get("allowed_symbols") or [],
            "trade_execution": "disabled",
            "human_review_required": True,
        },
        "risk_tolerance": request.context.risk_tolerance,
        "investment_horizon": request.context.investment_horizon,
    }


def _validate_portfolio_rebalance_payload(payload: RouteAgentResponseOutput, request: AgentQueryRequest, *, validate_message: bool = True) -> RouteAgentResponseOutput:
    payload.result_type = "portfolio_rebalance"
    result = dict(payload.result or {})
    result.setdefault("constraints", {})
    constraints = dict(result["constraints"] or {})
    constraints["trade_execution"] = "disabled"
    constraints["human_review_required"] = True
    result["constraints"] = constraints
    result["human_review_required"] = True
    result.setdefault("cash_target_weight", 0.0)
    result.setdefault("message", payload.message)
    changes = result.get("changes") or []
    normalized_changes = []
    seen_symbols = set()
    for change in changes:
        if not isinstance(change, dict):
            continue
        symbol = str(change.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        current = _as_float(change.get("current_weight"), 0.0)
        target = _as_float(change.get("target_weight"), current)
        normalized_changes.append({
            "symbol": symbol,
            "current_weight": round(current, 2),
            "target_weight": round(target, 2),
            "change": round(target - current, 2),
        })
        seen_symbols.add(symbol)
    for holding in (_portfolio_context(request).get("holdings") or []):
        if not isinstance(holding, dict):
            continue
        symbol = str(holding.get("symbol") or "").upper().strip()
        if not symbol or symbol in seen_symbols:
            continue
        current = _as_float(holding.get("weight"), 0.0)
        normalized_changes.append({"symbol": symbol, "current_weight": round(current, 2), "target_weight": round(current, 2), "change": 0.0})
    result["changes"] = normalized_changes
    PortfolioRebalanceResult.model_validate(result)
    if validate_message and _invalid_portfolio_message(payload.message):
        raise ValueError("portfolio rebalance agent message contains unsupported raw fields, dates, or execution language")
    if _contains_trade_execution_language(payload.message):
        raise ValueError("portfolio rebalance agent message implies trade execution")
    payload.result = result
    return payload


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _contains_trade_execution_language(message: str) -> bool:
    return bool(re.search(r"\b(executed|placed|submitted|bought|sold|trade executed|order placed)\b", message, flags=re.IGNORECASE))


def _invalid_portfolio_message(message: str) -> bool:
    raw_terms = r"\b(final_score|risk_prob|RVOL20|RSI14|pred_a|source_refs|as_of|Datetime)\b"
    date_pattern = r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:Z)?)?\b"
    return bool(re.search(raw_terms, message, flags=re.IGNORECASE) or re.search(date_pattern, message) or _contains_trade_execution_language(message))


def _market_brief_limit(message: str) -> int:
    return _market_screen_limit(message, default=5)


def _market_screen_limit(message: str, default: int = 5) -> int:
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
    return default


def _validate_agent_market_message(message: str, leaders: list[dict[str, Any]], *, check_symbols: bool = True) -> None:
    if not leaders:
        return
    if _contains_false_missing_market_data_claim(message):
        raise ValueError("agent market message claims tool data is missing despite grounded rows")
    if check_symbols and not _message_matches_leaders(message, leaders):
        raise ValueError("agent market message mentions symbols outside grounded tool rows")
    if _contains_user_facing_date(message):
        raise ValueError("agent market message contains raw dates/freshness fields")
    if "Not financial advice" in message:
        raise ValueError("agent market message contains removed disclaimer")


def _contains_false_missing_market_data_claim(message: str) -> bool:
    patterns = [
        r"\bno tool data\b",
        r"\bno grounded\b",
        r"\bno market data\b",
        r"\bno data returned\b",
        r"\bno .*list can be produced\b",
        r"\bcannot produce\b.*\b(?:top[- ]stocks|market)\b",
    ]
    return any(re.search(pattern, message, flags=re.IGNORECASE) for pattern in patterns)


def _message_matches_leaders(message: str, leaders: list[dict[str, Any]]) -> bool:
    leader_symbols = {_row_symbol(row) for row in leaders}
    leader_symbols.discard("")
    if not leader_symbols:
        return True
    mentioned = set(re.findall(r"\b[A-Z]{2,5}\b", message))
    ignored = {
        "ORCA", "RSI", "RSI14", "RVOL", "RVOL20",
        "AI", "API", "CEO", "CFO", "EPS", "ETF", "GDP", "CPI", "FOMC",
        "US", "USA", "USD", "MBR", "S3", "SQL", "LLM",
    }
    mentioned -= ignored
    return not mentioned or mentioned.issubset(leader_symbols)


def _contains_user_facing_date(message: str) -> bool:
    return bool(re.search(r"\b(?:as_of|Datetime|latest as_of|as of)\b|\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:Z)?)?\b", message, flags=re.IGNORECASE))


_SLIM_KEEP = frozenset({
    "Symbol", "symbol", "rank", "latest_price", "RSI14", "RVOL20",
    "risk_prob", "final_score", "sentiment_label", "sentiment_score",
    "article_count", "top_drivers", "as_of", "Datetime", "Close",
})


def _slim_market_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if k in _SLIM_KEEP} for row in rows]


def _row_symbol(row: dict[str, Any]) -> str:
    return str(row.get("symbol") or row.get("Symbol") or "").upper()


def _row_warnings(row: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not _row_symbol(row):
        warnings.append("missing symbol")
    if not (row.get("as_of") or row.get("Datetime")):
        warnings.append("missing data date")
    if row.get("final_score") is None:
        warnings.append("missing score")
    if row.get("latest_price") is None and row.get("Close") is None:
        warnings.append("missing price")
    return warnings


def _symbol_comparison_message(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No comparison rows returned from ORCA data source. Check data diagnostics or run EOD pipeline. Not financial advice."
    ranked = [row for row in rows if row.get("symbol")]
    order = ", ".join(f"{row['rank']}) {row['symbol']}" for row in ranked)
    strongest = ranked[0]
    weakest = ranked[-1]
    data_date = _latest_as_of(rows)
    warnings = [warning for row in rows for warning in (row.get("warnings") or [])]
    freshness = f" Data date: {data_date}." if data_date else " Data date unavailable."
    warning_text = f" Warnings: {'; '.join(warnings[:3])}." if warnings else ""
    return (
        f"Ranked read: {order}.{freshness}{warning_text} "
        f"Strongest: {strongest['symbol']} with score {_fmt_score(strongest.get('final_score'))}, "
        f"price {_fmt_metric(strongest.get('latest_price'))}, momentum {_fmt_metric(strongest.get('RSI14'))}, "
        f"trading activity {_fmt_metric(strongest.get('RVOL20'))}, and model risk {_fmt_percent(strongest.get('risk_prob'))}. "
        f"Weakest: {weakest['symbol']} with score {_fmt_score(weakest.get('final_score'))} and model risk {_fmt_percent(weakest.get('risk_prob'))}. "
        "Actionable read: favor strongest score/risk balance, but verify freshness and portfolio fit before acting. Not financial advice."
    )


def _symbol_comparison_message_is_grounded(message: str, rows: list[dict[str, Any]]) -> bool:
    if not message or not rows:
        return False
    symbols = {_row_symbol(row) for row in rows if _row_symbol(row)}
    mentioned = set(re.findall(r"\b[A-Z]{2,5}\b", message)) - {
        "ORCA", "RSI", "RVOL", "AI", "API", "EPS", "ETF", "GDP", "CPI", "USD"
    }
    if mentioned and not mentioned.issubset(symbols):
        return False
    row_dates = {str(row.get("as_of") or "")[:10] for row in rows if row.get("as_of")}
    message_dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", message))
    if message_dates and not message_dates.issubset(row_dates):
        return False
    return True


def _fmt_metric(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_score(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if numeric <= 1:
        return f"{numeric * 100:.1f}/100"
    return f"{numeric:.2f}"


def _fmt_percent(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


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
