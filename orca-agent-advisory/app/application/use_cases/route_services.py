from dataclasses import dataclass
from datetime import date
from typing import Any

from app.application.ports.backtest_provider import BacktestProvider, BacktestRequest
from app.application.ports.market_screen_provider import MarketScreenProvider
from app.application.ports.portfolio_provider import PortfolioProvider
from app.schemas.agent import AgentQueryRequest, AgentQueryResponse, RoutedAgentQuery, SuggestedAction
from app.schemas.enums import AgentRoute
from app.schemas.portfolio import PortfolioAccountSnapshot
from app.schemas.route_results import BacktestAnalysisResult, ComparisonRow, DataDiagnosticsResult, MarketBriefResult, PortfolioRebalanceChange, PortfolioRebalanceResult, ScreenCandidate, SymbolComparisonResult, TopStocksResult, UniverseScreenResult, WatchlistItem, WatchlistReviewResult


@dataclass
class AgentRouteServices:
    market_screen_provider: MarketScreenProvider
    backtest_provider: BacktestProvider | None = None
    portfolio_provider: PortfolioProvider | None = None

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

    def top_stocks(self, request: AgentQueryRequest, route: RoutedAgentQuery) -> AgentQueryResponse:
        stocks = [_screen_candidate(row) for row in self.market_screen_provider.screen_latest(_requested_count(request.message))]
        result = TopStocksResult(stocks=stocks)
        response = _response(route, "top_stocks", result.model_dump())
        response.message = "Top ORCA-ranked stocks loaded."
        return response

    def data_diagnostics(self, route: RoutedAgentQuery) -> AgentQueryResponse:
        result = DataDiagnosticsResult(diagnostics=self.market_screen_provider.diagnose())
        return _response(route, "data_diagnostics", result.model_dump())

    def portfolio_rebalance(self, request: AgentQueryRequest, route: RoutedAgentQuery) -> AgentQueryResponse:
        metadata = request.context.metadata
        constraints = _portfolio_constraints(metadata)
        has_account_id = bool(str(metadata.get("account_id") or "").strip())
        snapshot = _snapshot_from_provider(self.portfolio_provider, metadata)
        holdings = _holdings_from_snapshot(snapshot) if snapshot is not None else ([] if has_account_id else _holdings_from_context(request))
        if not holdings:
            return AgentQueryResponse(
                route=route.route,
                status="immediate",
                message="Portfolio rebalance needs holdings in context.metadata.portfolio or context.metadata.holdings. Format: [{'symbol':'AAPL','weight':25}, {'symbol':'MSFT','weight':35}]. No trades executed.",
                symbols=[] if has_account_id else route.symbols,
                result_type="portfolio_rebalance",
                result=PortfolioRebalanceResult(message="Missing portfolio holdings. Provide symbols with current weights.", constraints=constraints, human_review_required=True).model_dump(),
                suggested_actions=route.suggested_actions,
                router_confidence=route.confidence,
            )
        excluded = set(constraints["excluded_symbols"])
        allowed = set(constraints["allowed_symbols"])
        holdings = [holding for holding in holdings if holding["symbol"] not in excluded and (not allowed or holding["symbol"] in allowed)]
        if not holdings:
            result = PortfolioRebalanceResult(changes=[], cash_target_weight=100.0, constraints=constraints, human_review_required=True, message="No eligible portfolio holdings after constraints. No trades executed.")
            return AgentQueryResponse(route=route.route, status="immediate", message=result.message, symbols=[], result_type="portfolio_rebalance", result=result.model_dump(), suggested_actions=route.suggested_actions, router_confidence=route.confidence)
        symbols = [holding["symbol"] for holding in holdings]
        max_weight = float(constraints["max_single_asset_weight"])
        min_cash = float(constraints["min_cash_weight"])
        target = min(max_weight, round((100.0 - min_cash) / len(symbols), 2))
        total_target = round(target * len(symbols), 2)
        cash = round(max(min_cash, 100.0 - total_target), 2)
        changes = [PortfolioRebalanceChange(symbol=holding["symbol"], current_weight=holding["weight"], target_weight=target, change=round(target - holding["weight"], 2)) for holding in holdings]
        result = PortfolioRebalanceResult(
            changes=changes,
            cash_target_weight=cash,
            constraints=constraints,
            human_review_required=True,
            message="Deterministic planning-only rebalance. Review before any action; ORCA backend does not execute trades.",
        )
        return AgentQueryResponse(route=route.route, status="immediate", message=result.message, symbols=symbols, result_type="portfolio_rebalance", result=result.model_dump(), suggested_actions=route.suggested_actions, router_confidence=route.confidence)

    def backtest_analysis(self, request: AgentQueryRequest, route: RoutedAgentQuery) -> AgentQueryResponse:
        metadata = request.context.metadata
        symbols = route.symbols or request.context.symbols or ([request.context.symbol] if request.context.symbol else [])
        symbols = [str(symbol).strip().upper().replace(".", "-") for symbol in symbols if str(symbol).strip()]
        max_symbols = _positive_int(metadata.get("max_symbols"), 25)
        max_days = _positive_int(metadata.get("max_date_range_days"), 365)
        start_date = _str_or_none(metadata.get("start_date"))
        end_date = _str_or_none(metadata.get("end_date"))
        strategy = str(metadata.get("strategy") or "ORCA signal review")
        warnings: list[str] = []
        if len(symbols) > max_symbols:
            symbols = symbols[:max_symbols]
            warnings.append(f"symbol cap applied: max_symbols={max_symbols}")
        days = _date_range_days(start_date, end_date)
        if days is not None and days > max_days:
            result = BacktestAnalysisResult(
                backtest_spec={"symbols": symbols, "start_date": start_date, "end_date": end_date, "strategy": strategy, "data_source": "iceberg_spark"},
                status="disabled",
                limitation=f"Requested date range exceeds max_date_range_days={max_days}.",
                suggested_next_action="Narrow context.metadata.start_date/end_date or submit heavy job through /api/v1/query.",
                warnings=warnings + [f"date range cap rejected: requested_days={days}, max_date_range_days={max_days}"],
            )
            return AgentQueryResponse(route=route.route, status="immediate", message="Backtest analysis rejected by production safety caps. No chart rendered.", symbols=symbols, result_type="backtest_analysis", result=result.model_dump(), suggested_actions=route.suggested_actions, router_confidence=route.confidence)
        spec = {"symbols": symbols, "start_date": start_date, "end_date": end_date, "strategy": strategy, "data_source": "iceberg_spark"}
        if self.backtest_provider is not None and self.backtest_provider.is_available():
            provider_result = self.backtest_provider.run_backtest(BacktestRequest(symbols=symbols, start_date=start_date, end_date=end_date, strategy=strategy, metadata=metadata))
            result = BacktestAnalysisResult(
                backtest_spec=spec,
                status="completed",
                limitation="Iceberg/Spark backtest adapter returned deterministic API-safe summary. Chart rendering disabled in API.",
                suggested_next_action="For heavy workloads, use /api/v1/query.",
                metrics=provider_result.metrics,
                trades_summary=provider_result.trades_summary,
                equity_curve_sampled=provider_result.equity_curve_sampled,
                warnings=warnings + provider_result.warnings,
            )
            return AgentQueryResponse(route=route.route, status="immediate", message="Backtest analysis completed from Iceberg/Spark provider. No external market data fetched.", symbols=symbols, result_type="backtest_analysis", result=result.model_dump(), suggested_actions=route.suggested_actions, router_confidence=route.confidence)
        result = BacktestAnalysisResult(
            backtest_spec=spec,
            status="planned",
            limitation="Iceberg/Spark backtest provider is not configured or unavailable. External market data calls are disabled in ORCA API for production safety.",
            suggested_next_action="Configure Iceberg-backed backtest adapter or submit planned heavy workload through /api/v1/query.",
            warnings=warnings,
        )
        return AgentQueryResponse(route=route.route, status="immediate", message="Backtest analysis planned only. No external market data fetched and no chart rendered.", symbols=symbols, result_type="backtest_analysis", result=result.model_dump(), suggested_actions=route.suggested_actions, router_confidence=route.confidence)


def _response(route: RoutedAgentQuery, result_type: str, result: dict[str, Any]) -> AgentQueryResponse:
    return AgentQueryResponse(route=route.route, status="immediate", message=route.message, symbols=route.symbols, result_type=result_type, result=result, suggested_actions=route.suggested_actions or [SuggestedAction(label="Ask for single-symbol advisory", route=AgentRoute.SINGLE_SYMBOL_ADVISORY)], router_confidence=route.confidence)


def _requested_count(message: str, default: int = 10) -> int:
    import re

    text = str(message).lower()
    match = re.search(r"\b(?:top\s*)?(\d{1,2})\s*(?:stocks?|names?|tickers?)\b", text)
    if match:
        return max(1, min(int(match.group(1)), 20))
    return default


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


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _date_range_days(start_date: str | None, end_date: str | None) -> int | None:
    if not start_date or not end_date:
        return None
    try:
        return (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1
    except ValueError:
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


def _snapshot_from_provider(provider: PortfolioProvider | None, metadata: dict[str, Any]) -> PortfolioAccountSnapshot | None:
    if provider is None:
        return None
    account_id = str(metadata.get("account_id") or "").strip()
    if not account_id:
        return None
    tenant_id = _str_or_none(metadata.get("tenant_id"))
    return provider.get_account_snapshot(account_id=account_id, tenant_id=tenant_id)


def _holdings_from_snapshot(snapshot: PortfolioAccountSnapshot) -> list[dict[str, Any]]:
    holdings: list[dict[str, Any]] = []
    total_value = sum(position.market_value or 0.0 for position in snapshot.positions) + max(snapshot.cash, 0.0)
    for position in snapshot.positions:
        symbol = position.symbol.strip().upper().replace(".", "-")
        weight = position.weight
        if weight is None and position.market_value is not None and total_value > 0:
            weight = (position.market_value / total_value) * 100.0
        if symbol and weight is not None:
            holdings.append({"symbol": symbol, "weight": round(float(weight), 2)})
    return holdings


def _portfolio_constraints(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_single_asset_weight": _float(metadata.get("max_single_asset_weight")) or 40.0,
        "min_cash_weight": _float(metadata.get("min_cash_weight")) or 0.0,
        "excluded_symbols": _symbol_list(metadata.get("excluded_symbols")),
        "allowed_symbols": _symbol_list(metadata.get("allowed_symbols")),
        "target_method": "equal_weight_capped",
        "trade_execution": "disabled",
        "human_review_required": True,
    }


def _symbol_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().upper().replace(".", "-") for item in value if str(item).strip()]
