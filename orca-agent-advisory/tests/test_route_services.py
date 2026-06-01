from app.application.use_cases.route_services import AgentRouteServices
from app.application.ports.backtest_provider import BacktestProviderResult, BacktestRequest
from app.schemas.agent import AgentQueryRequest, AgentContext, RoutedAgentQuery
from app.schemas.enums import AgentRoute


class FakeMarketScreenProvider:
    def load_symbols(self, symbols: list[str]) -> list[dict]:
        return [
            {"Symbol": "AAA", "final_score": 0.9, "pred_a": "UP", "Close": 10.5, "r1": 0.01, "RVOL20": 1.2, "RSI14": 55, "risk_prob": 0.2, "Datetime": "2026-01-02"},
            {"symbol": "BBB", "final_score": None, "predicted_direction": None, "latest_price": None},
        ]

    def screen_latest(self, limit: int = 10) -> list[dict]:
        return [
            {"Symbol": "CCC", "final_score": 0.8, "pred_a": "UP", "latest_price": 20, "as_of": "2026-01-03"},
            {"Symbol": "DDD", "final_score": None},
        ][:limit]

    def diagnose(self) -> dict:
        return {"status": "ok"}


class FakeBacktestProvider:
    def is_available(self) -> bool:
        return True

    def run_backtest(self, request: BacktestRequest) -> BacktestProviderResult:
        return BacktestProviderResult(
            metrics={"total_return": 0.12, "symbols": request.symbols},
            trades_summary={"trades": 2, "strategy": request.strategy},
            equity_curve_sampled=[{"date": request.start_date, "equity": 1.0}, {"date": request.end_date, "equity": 1.12}],
        )


def _route(route: AgentRoute, symbols: list[str] | None = None) -> RoutedAgentQuery:
    return RoutedAgentQuery(route=route, confidence=0.8, symbols=symbols or ["AAA", "BBB"], message="test")


def test_symbol_comparison_enriches_rows_and_tolerates_missing_values() -> None:
    response = AgentRouteServices(FakeMarketScreenProvider()).symbol_comparison(_route(AgentRoute.SYMBOL_COMPARISON))

    rows = response.result["rows"]
    assert rows[0]["symbol"] == "AAA"
    assert rows[0]["latest_price"] == 10.5
    assert rows[0]["r1"] == 0.01
    assert rows[0]["RVOL20"] == 1.2
    assert rows[0]["RSI14"] == 55.0
    assert rows[0]["risk_prob"] == 0.2
    assert rows[0]["as_of"] == "2026-01-02"
    assert rows[1]["status"] == "warning"
    assert "missing latest_price" in rows[1]["warnings"]


def test_watchlist_review_includes_enriched_fields_for_null_columns() -> None:
    response = AgentRouteServices(FakeMarketScreenProvider()).watchlist_review(_route(AgentRoute.WATCHLIST_REVIEW))

    items = response.result["items"]
    assert items[0]["latest_price"] == 10.5
    assert items[0]["predicted_direction"] == "UP"
    assert items[1]["final_score"] is None
    assert items[1]["status"] == "warning"
    assert "missing as_of" in items[1]["warnings"]


def test_market_brief_leaders_include_enriched_fields_and_warnings() -> None:
    response = AgentRouteServices(FakeMarketScreenProvider()).market_brief(_route(AgentRoute.MARKET_BRIEF, []))

    leaders = response.result["leaders"]
    assert leaders[0]["symbol"] == "CCC"
    assert leaders[0]["latest_price"] == 20.0
    assert leaders[0]["as_of"] == "2026-01-03"
    assert leaders[1]["status"] == "warning"
    assert "missing latest_price" in leaders[1]["warnings"]


def test_backtest_unavailable_returns_planned_response() -> None:
    request = AgentQueryRequest(message="backtest", context=AgentContext(symbols=["AAA"], metadata={"start_date": "2026-01-01", "end_date": "2026-01-31"}))

    response = AgentRouteServices(FakeMarketScreenProvider()).backtest_analysis(request, _route(AgentRoute.BACKTEST_ANALYSIS, ["AAA"]))

    assert response.result["status"] == "planned"
    assert response.result["metrics"] is None
    assert "External market data calls are disabled" in response.result["limitation"]


def test_backtest_provider_returns_metrics() -> None:
    request = AgentQueryRequest(message="backtest", context=AgentContext(symbols=["AAA"], metadata={"start_date": "2026-01-01", "end_date": "2026-01-31", "strategy": "buy_hold"}))

    response = AgentRouteServices(FakeMarketScreenProvider(), FakeBacktestProvider()).backtest_analysis(request, _route(AgentRoute.BACKTEST_ANALYSIS, ["AAA"]))

    assert response.result["status"] == "completed"
    assert response.result["metrics"] == {"total_return": 0.12, "symbols": ["AAA"]}
    assert response.result["trades_summary"] == {"trades": 2, "strategy": "buy_hold"}
    assert response.result["equity_curve_sampled"][-1]["equity"] == 1.12


def test_backtest_caps_limit_symbols_and_reject_date_range() -> None:
    request = AgentQueryRequest(message="backtest", context=AgentContext(symbols=["AAA", "BBB", "CCC"], metadata={"start_date": "2026-01-01", "end_date": "2026-02-15", "max_symbols": 2, "max_date_range_days": 10}))

    response = AgentRouteServices(FakeMarketScreenProvider(), FakeBacktestProvider()).backtest_analysis(request, _route(AgentRoute.BACKTEST_ANALYSIS, ["AAA", "BBB", "CCC"]))

    assert response.result["status"] == "disabled"
    assert response.symbols == ["AAA", "BBB"]
    assert "symbol cap applied" in response.result["warnings"][0]
    assert "date range cap rejected" in response.result["warnings"][1]


def test_backtest_code_does_not_import_yfinance() -> None:
    import pathlib

    app_dir = pathlib.Path(__file__).parents[1] / "app"
    matches = [path for path in app_dir.rglob("*.py") if "import yfinance" in path.read_text(encoding="utf-8") or "from yfinance" in path.read_text(encoding="utf-8")]
    assert matches == []
