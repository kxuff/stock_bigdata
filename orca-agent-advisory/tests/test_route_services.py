from app.application.use_cases.route_services import AgentRouteServices
from app.application.services.agent_query_router_service import AgentQueryRouterService, _extract_symbols
from app.application.use_cases.streaming_route_services import StreamingRouteServices
from app.application.ports.backtest_provider import BacktestProviderResult, BacktestRequest
from app.application.ports.portfolio_provider import InMemoryPortfolioProvider
from app.infrastructure.bigdata.streaming_providers import KafkaTopicMetadataInspectionProvider
from app.schemas.agent import AgentQueryRequest, AgentContext, RoutedAgentQuery
from app.schemas.enums import AgentRoute
from app.schemas.portfolio import PortfolioAccountSnapshot, PortfolioPosition


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


class FakeStreamingProvider:
    def get_pipeline_health(self, lookback_minutes: int) -> list[dict]:
        return []

    def get_symbol_freshness(self, symbols: list[str], lookback_minutes: int) -> list[dict]:
        return []

    def get_ingestion_lag(self, lookback_minutes: int) -> list[dict]:
        return []

    def get_latest_alerts(self, symbols: list[str], severities: list[str], limit: int, lookback_minutes: int) -> list[dict]:
        return []

    def get_active_symbol_alerts(self, symbol: str, lookback_minutes: int) -> list[dict]:
        return []

    def find_quality_incidents(self, symbols: list[str], lookback_minutes: int, limit: int) -> list[dict]:
        return []

    def compare_streaming_to_batch_features(self, symbols: list[str], as_of_date: str | None) -> list[dict]:
        return []

    def inspect_topics(self) -> list[dict]:
        return [{"topic": "stock-market", "status": "diagnostic_only", "sample": {}, "limitation": "Kafka direct topic inspection not enabled; use Iceberg streaming tables for read-only diagnostics."}]


class FakeKafkaInspectionProvider:
    def inspect_topics(self, topics: list[str] | None = None) -> list[dict]:
        return [{"topic": "stock-market", "status": "ok", "partition_count": 2, "latest_offsets": {"0": 10, "1": 20}, "consumer_lag": {"0": 1, "1": 0}, "sample": {}}]


class FailingKafkaInspectionProvider:
    def inspect_topics(self, topics: list[str] | None = None) -> list[dict]:
        raise TimeoutError("metadata timeout")


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


def test_symbol_extraction_ignores_orca_product_name() -> None:
    assert _extract_symbols("Compare AAPL and MSFT using latest ORCA predictions") == ["AAPL", "MSFT"]


def test_router_fallback_resolves_symbol_from_history() -> None:
    request = AgentQueryRequest(message="what about it?", history=[{"role": "user", "content": "Analyze AAPL", "metadata": {"symbol": "AAPL"}}])

    routed = AgentQueryRouterService().route(request)

    assert routed.route == AgentRoute.SINGLE_SYMBOL_ADVISORY
    assert routed.symbols == ["AAPL"]


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


def test_streaming_topic_inspection_uses_kafka_provider_metadata() -> None:
    streaming = FakeStreamingProvider()
    response = StreamingRouteServices(streaming, streaming, streaming, topic_inspection_provider=FakeKafkaInspectionProvider()).topic_inspection(_route(AgentRoute.STREAMING_TOPIC_INSPECTION, []))

    sample = response.result["samples"][0]
    assert sample["topic"] == "stock-market"
    assert sample["status"] == "ok"
    assert sample["partition_count"] == 2
    assert sample["latest_offsets"] == {"0": 10, "1": 20}
    assert sample["consumer_lag"] == {"0": 1, "1": 0}
    assert sample["sample"] == {}


def test_streaming_topic_inspection_falls_back_to_diagnostic_provider() -> None:
    streaming = FakeStreamingProvider()
    response = StreamingRouteServices(streaming, streaming, streaming).topic_inspection(_route(AgentRoute.STREAMING_TOPIC_INSPECTION, []))

    assert response.result["samples"][0]["status"] == "diagnostic_only"
    assert "Kafka direct topic inspection not enabled" in response.result["samples"][0]["limitation"]


def test_streaming_topic_inspection_failure_soft() -> None:
    streaming = FakeStreamingProvider()
    response = StreamingRouteServices(streaming, streaming, streaming, topic_inspection_provider=FailingKafkaInspectionProvider()).topic_inspection(_route(AgentRoute.STREAMING_TOPIC_INSPECTION, []))

    sample = response.result["samples"][0]
    assert sample["status"] == "error"
    assert sample["error"] == "metadata timeout"
    assert sample["limitation"] == "Kafka direct topic inspection failed soft."


def test_kafka_topic_provider_rejects_topics_outside_allowlist_before_client_import() -> None:
    provider = KafkaTopicMetadataInspectionProvider(bootstrap_servers="localhost:9092", allowed_topics=["stock-market"])

    rows = provider.inspect_topics(["secret-topic"])

    assert rows == [{"topic": "secret-topic", "status": "rejected", "error": "Topic not in Kafka inspection allowlist."}]


def test_kafka_topic_provider_sample_masked_when_enabled() -> None:
    provider = KafkaTopicMetadataInspectionProvider(bootstrap_servers="localhost:9092", allowed_topics=["stock-market"], sample_enabled=True, sample_max_bytes=4)

    sample = provider._sample_disabled()

    assert sample["status"] == "disabled"
    assert "metadata inspection only" in sample["limitation"]


def test_portfolio_rebalance_uses_provider_snapshot_when_account_id_present() -> None:
    provider = InMemoryPortfolioProvider({
        "acct-1": PortfolioAccountSnapshot(
            account_id="acct-1",
            tenant_id="tenant-a",
            base_currency="USD",
            positions=[PortfolioPosition(symbol="AAA", weight=70), PortfolioPosition(symbol="BBB", market_value=20)],
            cash=10,
            as_of="2026-01-02T00:00:00Z",
            source="test",
        )
    })
    request = AgentQueryRequest(message="rebalance", context=AgentContext(metadata={"account_id": "acct-1", "tenant_id": "tenant-a"}))

    response = AgentRouteServices(FakeMarketScreenProvider(), portfolio_provider=provider).portfolio_rebalance(request, _route(AgentRoute.PORTFOLIO_REBALANCE, []))

    assert response.symbols == ["AAA", "BBB"]
    assert response.result["changes"][0]["current_weight"] == 70.0
    assert response.result["constraints"]["trade_execution"] == "disabled"
    assert response.result["human_review_required"] is True


def test_portfolio_rebalance_tenant_mismatch_does_not_leak_provider_or_context() -> None:
    provider = InMemoryPortfolioProvider({
        "acct-1": PortfolioAccountSnapshot(
            account_id="acct-1",
            tenant_id="tenant-a",
            base_currency="USD",
            positions=[PortfolioPosition(symbol="SECRET", weight=100)],
            cash=0,
            as_of="2026-01-02T00:00:00Z",
            source="test",
        )
    })
    request = AgentQueryRequest(message="rebalance", context=AgentContext(metadata={"account_id": "acct-1", "tenant_id": "tenant-b", "holdings": [{"symbol": "AAA", "weight": 50}]}))

    response = AgentRouteServices(FakeMarketScreenProvider(), portfolio_provider=provider).portfolio_rebalance(request, _route(AgentRoute.PORTFOLIO_REBALANCE, []))

    assert response.symbols == []
    assert response.result["changes"] == []
    assert "SECRET" not in str(response.result)


def test_portfolio_rebalance_empty_or_malformed_portfolio_is_safe() -> None:
    request = AgentQueryRequest(message="rebalance", context=AgentContext(metadata={"portfolio": {"positions": [{"symbol": "AAA"}, "bad"]}}))

    response = AgentRouteServices(FakeMarketScreenProvider()).portfolio_rebalance(request, _route(AgentRoute.PORTFOLIO_REBALANCE, []))

    assert response.result["changes"] == []
    assert response.result["human_review_required"] is True
    assert response.result["constraints"]["trade_execution"] == "disabled"


def test_portfolio_rebalance_applies_constraints_and_excluded_symbols() -> None:
    request = AgentQueryRequest(
        message="rebalance",
        context=AgentContext(metadata={"holdings": [{"symbol": "AAA", "weight": 80}, {"symbol": "BBB", "weight": 10}, {"symbol": "CCC", "weight": 10}], "excluded_symbols": ["BBB"], "allowed_symbols": ["AAA", "CCC"], "max_single_asset_weight": 30, "min_cash_weight": 20}),
    )

    response = AgentRouteServices(FakeMarketScreenProvider()).portfolio_rebalance(request, _route(AgentRoute.PORTFOLIO_REBALANCE, []))

    assert response.symbols == ["AAA", "CCC"]
    assert [change["target_weight"] for change in response.result["changes"]] == [30.0, 30.0]
    assert response.result["cash_target_weight"] == 40.0
    assert response.result["constraints"]["excluded_symbols"] == ["BBB"]


def test_portfolio_rebalance_never_sets_trade_execution_enabled() -> None:
    request = AgentQueryRequest(message="rebalance", context=AgentContext(metadata={"holdings": [{"symbol": "AAA", "weight": 100}], "trade_execution": "enabled"}))

    response = AgentRouteServices(FakeMarketScreenProvider()).portfolio_rebalance(request, _route(AgentRoute.PORTFOLIO_REBALANCE, []))

    assert response.result["constraints"]["trade_execution"] == "disabled"
    assert response.result["human_review_required"] is True
