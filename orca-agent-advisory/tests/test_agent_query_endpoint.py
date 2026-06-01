import sys
import types

from fastapi.testclient import TestClient

psycopg_stub = types.ModuleType("psycopg")
psycopg_stub.errors = types.SimpleNamespace(UniqueViolation=Exception)
psycopg_rows_stub = types.ModuleType("psycopg.rows")
psycopg_rows_stub.dict_row = object()
redis_stub = types.ModuleType("redis")
redis_stub.Redis = type("Redis", (), {"from_url": staticmethod(lambda url: object())})
rq_stub = types.ModuleType("rq")
rq_stub.Queue = type("Queue", (), {})
sys.modules.setdefault("psycopg", psycopg_stub)
sys.modules.setdefault("psycopg.rows", psycopg_rows_stub)
sys.modules.setdefault("redis", redis_stub)
sys.modules.setdefault("rq", rq_stub)

from app.infrastructure.storage.agent_route_audit_store import AgentRouteAuditEntry
from app.main import app, get_agent_route_audit_store, get_autonomous_agent_service
from app.schemas.agent import AgentQueryResponse
from app.schemas.enums import AgentRoute


class FakeAutonomousAgentService:
    def query(self, request):
        route = AgentRoute(request.context.metadata["route"])
        return AgentQueryResponse(
            route=route,
            status="immediate",
            message=f"fake {route.value}",
            symbols=request.context.symbols,
            result_type=route.value if route not in {AgentRoute.CLARIFICATION, AgentRoute.OUT_OF_SCOPE} else None,
            result={"route": route.value},
            router_confidence=0.91,
        )


class FailingAutonomousAgentService:
    def query(self, request):
        raise RuntimeError("agent service failed")


class FakeAuditStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.entries: list[AgentRouteAuditEntry] = []

    def record(self, entry: AgentRouteAuditEntry) -> None:
        if self.fail:
            raise RuntimeError("audit failed")
        self.entries.append(entry)


def _client(service) -> TestClient:
    app.dependency_overrides[get_autonomous_agent_service] = lambda: service
    return TestClient(app)


def _post_route(route: AgentRoute, symbols: list[str] | None = None):
    client = _client(FakeAutonomousAgentService())
    try:
        return client.post(
            "/api/v1/agent/query",
            json={
                "message": f"test {route.value}",
                "context": {
                    "symbols": symbols or [],
                    "metadata": {"route": route.value},
                },
            },
        )
    finally:
        app.dependency_overrides.clear()


def test_agent_query_returns_clarification_route() -> None:
    response = _post_route(AgentRoute.CLARIFICATION)

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "clarification"
    assert payload["status"] == "immediate"
    assert payload["result_type"] is None


def test_agent_query_returns_out_of_scope_route() -> None:
    response = _post_route(AgentRoute.OUT_OF_SCOPE)

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "out_of_scope"
    assert payload["result_type"] is None


def test_agent_query_returns_symbol_comparison_route() -> None:
    response = _post_route(AgentRoute.SYMBOL_COMPARISON, ["AAPL", "MSFT"])

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "symbol_comparison"
    assert payload["symbols"] == ["AAPL", "MSFT"]
    assert payload["result_type"] == "symbol_comparison"


def test_agent_query_returns_portfolio_rebalance_route() -> None:
    response = _post_route(AgentRoute.PORTFOLIO_REBALANCE, ["AAPL", "MSFT", "NVDA"])

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "portfolio_rebalance"
    assert payload["symbols"] == ["AAPL", "MSFT", "NVDA"]
    assert payload["result_type"] == "portfolio_rebalance"


def test_agent_query_returns_backtest_analysis_route() -> None:
    response = _post_route(AgentRoute.BACKTEST_ANALYSIS, ["AAPL"])

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "backtest_analysis"
    assert payload["symbols"] == ["AAPL"]
    assert payload["result_type"] == "backtest_analysis"


def test_agent_query_rejects_extra_payload_field() -> None:
    client = _client(FakeAutonomousAgentService())
    try:
        response = client.post(
            "/api/v1/agent/query",
            json={
                "message": "compare AAPL and MSFT",
                "context": {"symbols": ["AAPL", "MSFT"]},
                "unexpected": True,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    payload = response.json()
    assert payload["status"] == "ERROR"
    assert payload["error_code"] == "INVALID_REQUEST"


def test_agent_query_service_exception_returns_500() -> None:
    app.dependency_overrides[get_autonomous_agent_service] = lambda: FailingAutonomousAgentService()
    client = TestClient(app, raise_server_exceptions=False)
    try:
        response = client.post("/api/v1/agent/query", json={"message": "compare AAPL and MSFT"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500


def test_agent_query_success_creates_audit_entry() -> None:
    audit_store = FakeAuditStore()
    app.dependency_overrides[get_autonomous_agent_service] = lambda: FakeAutonomousAgentService()
    app.dependency_overrides[get_agent_route_audit_store] = lambda: audit_store
    client = TestClient(app)
    try:
        response = client.post(
            "/api/v1/agent/query",
            headers={"X-Tenant-Id": "tenant-1", "X-User-Id": "user-1"},
            json={
                "message": "compare AAPL and MSFT",
                "context": {"symbols": ["AAPL", "MSFT"], "metadata": {"route": "symbol_comparison", "request_id": "req-1"}},
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert len(audit_store.entries) == 1
    entry = audit_store.entries[0]
    assert entry.request_id == "req-1"
    assert entry.tenant_id == "tenant-1"
    assert entry.user_id == "user-1"
    assert entry.message_hash != "compare AAPL and MSFT"
    assert len(entry.message_hash) == 64
    assert entry.route == "symbol_comparison"
    assert entry.router_confidence == 0.91
    assert entry.symbols == ["AAPL", "MSFT"]
    assert entry.status == "succeeded"
    assert entry.error_code is None


def test_agent_query_service_failure_creates_audit_entry() -> None:
    audit_store = FakeAuditStore()
    app.dependency_overrides[get_autonomous_agent_service] = lambda: FailingAutonomousAgentService()
    app.dependency_overrides[get_agent_route_audit_store] = lambda: audit_store
    client = TestClient(app, raise_server_exceptions=False)
    try:
        response = client.post("/api/v1/agent/query", json={"message": "compare AAPL and MSFT"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert len(audit_store.entries) == 1
    entry = audit_store.entries[0]
    assert entry.status == "failed"
    assert entry.error_code == "INTERNAL_ERROR"
    assert entry.route is None


def test_agent_query_audit_store_failure_does_not_break_response() -> None:
    app.dependency_overrides[get_autonomous_agent_service] = lambda: FakeAutonomousAgentService()
    app.dependency_overrides[get_agent_route_audit_store] = lambda: FakeAuditStore(fail=True)
    client = TestClient(app, raise_server_exceptions=False)
    try:
        response = client.post(
            "/api/v1/agent/query",
            json={"message": "compare AAPL and MSFT", "context": {"metadata": {"route": "symbol_comparison"}}},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
