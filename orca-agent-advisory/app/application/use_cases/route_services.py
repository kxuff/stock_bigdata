from dataclasses import dataclass
from typing import Any

from app.application.ports.market_screen_provider import MarketScreenProvider
from app.schemas.agent import AgentQueryRequest, AgentQueryResponse, RoutedAgentQuery, SuggestedAction
from app.schemas.enums import AgentRoute
from app.schemas.route_results import BacktestAnalysisResult, ComparisonRow, DataDiagnosticsResult, MarketBriefResult, PortfolioRebalanceChange, PortfolioRebalanceResult, ScreenCandidate, SymbolComparisonResult, UniverseScreenResult, WatchlistItem, WatchlistReviewResult


@dataclass
class AgentRouteServices:
    market_screen_provider: MarketScreenProvider

    def symbol_comparison(self, route: RoutedAgentQuery) -> AgentQueryResponse:
        rows = [_comparison_row(row, i + 1) for i, row in enumerate(sorted(self.market_screen_provider.load_symbols(route.symbols), key=lambda r: r.get("final_score") or 0, reverse=True))]
        result = SymbolComparisonResult(rows=rows)
        return _response(route, "symbol_comparison", result.model_dump())

    def watchlist_review(self, route: RoutedAgentQuery) -> AgentQueryResponse:
        result = WatchlistReviewResult(items=[_watchlist_item(row) for row in self.market_screen_provider.load_symbols(route.symbols)])
        return _response(route, "watchlist_review", result.model_dump())

    def universe_screen(self, route: RoutedAgentQuery) -> AgentQueryResponse:
        result = UniverseScreenResult(candidates=[_screen_candidate(row) for row in self.market_screen_provider.screen_latest(10)])
        return _response(route, "universe_screen", result.model_dump())

    def market_brief(self, route: RoutedAgentQuery) -> AgentQueryResponse:
        leaders = [_screen_candidate(row) for row in self.market_screen_provider.screen_latest(5)]
        result = MarketBriefResult(summary="Latest ORCA prediction leaders loaded from prediction table.", leaders=leaders)
        return _response(route, "market_brief", result.model_dump())

    def data_diagnostics(self, route: RoutedAgentQuery) -> AgentQueryResponse:
        result = DataDiagnosticsResult(diagnostics=self.market_screen_provider.diagnose())
        return _response(route, "data_diagnostics", result.model_dump())

    def portfolio_rebalance(self, request: AgentQueryRequest, route: RoutedAgentQuery) -> AgentQueryResponse:
        holdings = _holdings_from_context(request)
        if not holdings:
            return AgentQueryResponse(
                route=route.route,
                status="immediate",
                message="Portfolio rebalance needs holdings in context.metadata.portfolio or context.metadata.holdings. Format: [{'symbol':'AAPL','weight':25}, {'symbol':'MSFT','weight':35}]. No trades executed.",
                symbols=route.symbols,
                result_type="portfolio_rebalance",
                result=PortfolioRebalanceResult(message="Missing portfolio holdings. Provide symbols with current weights.", constraints={"max_single_asset_weight": 40.0}, human_review_required=True).model_dump(),
                suggested_actions=route.suggested_actions,
                router_confidence=route.confidence,
            )
        symbols = [holding["symbol"] for holding in holdings]
        target = min(40.0, round(100.0 / len(symbols), 2))
        total_target = round(target * len(symbols), 2)
        cash = round(max(0.0, 100.0 - total_target), 2)
        changes = [PortfolioRebalanceChange(symbol=holding["symbol"], current_weight=holding["weight"], target_weight=target, change=round(target - holding["weight"], 2)) for holding in holdings]
        result = PortfolioRebalanceResult(
            changes=changes,
            cash_target_weight=cash,
            constraints={"max_single_asset_weight": 40.0, "target_method": "equal_weight_capped", "trade_execution": "disabled"},
            human_review_required=True,
            message="Deterministic planning-only rebalance. Review before any action; ORCA backend does not execute trades.",
        )
        return AgentQueryResponse(route=route.route, status="immediate", message=result.message, symbols=symbols, result_type="portfolio_rebalance", result=result.model_dump(), suggested_actions=route.suggested_actions, router_confidence=route.confidence)

    def backtest_analysis(self, request: AgentQueryRequest, route: RoutedAgentQuery) -> AgentQueryResponse:
        symbols = route.symbols or request.context.symbols or ([request.context.symbol] if request.context.symbol else [])
        result = BacktestAnalysisResult(
            backtest_spec={
                "symbols": symbols,
                "start_date": request.context.metadata.get("start_date"),
                "end_date": request.context.metadata.get("end_date"),
                "strategy": request.context.metadata.get("strategy", "ORCA signal review"),
                "data_source": "not_connected_in_orca_api",
            },
            status="planned",
            limitation="Backend backtest service is not connected to Iceberg here and yfinance is disabled in ORCA API for production safety.",
            suggested_next_action="Use the Streamlit Stock Picks backtest page or connect an Iceberg-backed backtest adapter.",
        )
        return AgentQueryResponse(route=route.route, status="immediate", message="Backtest analysis route supported as planning/spec response. No external market data fetched.", symbols=symbols, result_type="backtest_analysis", result=result.model_dump(), suggested_actions=route.suggested_actions, router_confidence=route.confidence)


def _response(route: RoutedAgentQuery, result_type: str, result: dict[str, Any]) -> AgentQueryResponse:
    return AgentQueryResponse(route=route.route, status="immediate", message=route.message, symbols=route.symbols, result_type=result_type, result=result, suggested_actions=route.suggested_actions or [SuggestedAction(label="Ask for single-symbol advisory", route=AgentRoute.SINGLE_SYMBOL_ADVISORY)], router_confidence=route.confidence)


def _screen_candidate(row: dict[str, Any]) -> ScreenCandidate:
    warnings = _warnings(row)
    return ScreenCandidate(
        symbol=_symbol(row),
        final_score=_float(row.get("final_score")),
        predicted_direction=_str_or_none(row.get("pred_a") or row.get("predicted_direction")),
        as_of=_str_or_none(row.get("freshness") or row.get("as_of") or row.get("Datetime")),
        latest_price=_float(_first_present(row, "latest_price", "Close")),
        r1=_float(row.get("r1")),
        RVOL20=_float(row.get("RVOL20")),
        RSI14=_float(row.get("RSI14")),
        risk_prob=_float(row.get("risk_prob")),
        status="warning" if warnings else "ok",
        warnings=warnings,
    )


def _comparison_row(row: dict[str, Any], rank: int) -> ComparisonRow:
    c = _screen_candidate(row)
    return ComparisonRow(symbol=c.symbol, final_score=c.final_score, predicted_direction=c.predicted_direction, rank=rank, latest_price=c.latest_price, r1=c.r1, RVOL20=c.RVOL20, RSI14=c.RSI14, risk_prob=c.risk_prob, as_of=c.as_of, status=c.status, warnings=c.warnings)


def _watchlist_item(row: dict[str, Any]) -> WatchlistItem:
    c = _screen_candidate(row)
    return WatchlistItem(symbol=c.symbol, status=c.status or "reviewed", final_score=c.final_score, predicted_direction=c.predicted_direction, latest_price=c.latest_price, r1=c.r1, RVOL20=c.RVOL20, RSI14=c.RSI14, risk_prob=c.risk_prob, as_of=c.as_of, warnings=c.warnings)


def _symbol(row: dict[str, Any]) -> str:
    return str(row.get("Symbol") or row.get("symbol") or "")


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _str_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


def _warnings(row: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if _symbol(row) == "":
        warnings.append("missing symbol")
    if _first_present(row, "freshness", "as_of", "Datetime") is None:
        warnings.append("missing as_of")
    if _first_present(row, "latest_price", "Close") is None:
        warnings.append("missing latest_price")
    if row.get("final_score") is None:
        warnings.append("missing final_score")
    return warnings


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _holdings_from_context(request: AgentQueryRequest) -> list[dict[str, Any]]:
    raw = request.context.metadata.get("holdings") or request.context.metadata.get("portfolio")
    if isinstance(raw, dict):
        raw = raw.get("holdings") or raw.get("positions") or raw.get("assets")
    if not isinstance(raw, list):
        return []
    holdings: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or item.get("ticker") or "").strip().upper().replace(".", "-")
        weight = _float(item.get("weight") if item.get("weight") is not None else item.get("current_weight"))
        if symbol and weight is not None:
            holdings.append({"symbol": symbol, "weight": round(weight, 2)})
    return holdings
