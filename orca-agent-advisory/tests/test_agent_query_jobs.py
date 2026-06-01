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

from app.main import _jobs, _jobs_lock, _now_iso, app, get_autonomous_agent_service
from app.schemas.agent import AgentQueryResponse
from app.schemas.enums import AgentRoute


class FakeAutonomousAgentService:
    def query(self, request):
        route = AgentRoute(request.context.metadata.get("route", AgentRoute.SYMBOL_COMPARISON.value))
        return AgentQueryResponse(
            route=route,
            status="immediate",
            message=f"fake {route.value}",
            symbols=request.context.symbols,
            result_type=route.value,
            result={"route": route.value},
            router_confidence=0.91,
        )


class FailingAutonomousAgentService:
    def query(self, request):
        raise RuntimeError("agent service failed")


def setup_function() -> None:
    app.dependency_overrides.clear()
    with _jobs_lock:
        _jobs.clear()


def teardown_function() -> None:
    app.dependency_overrides.clear()
    with _jobs_lock:
        _jobs.clear()


def _client(service) -> TestClient:
    app.dependency_overrides[get_autonomous_agent_service] = lambda: service
    return TestClient(app, raise_server_exceptions=False)


def _query_payload(route: AgentRoute = AgentRoute.SYMBOL_COMPARISON) -> dict:
    return {
        "message": "compare AAPL and MSFT",
        "context": {
            "symbols": ["AAPL", "MSFT"],
            "metadata": {"route": route.value},
        },
    }


def test_agent_query_job_create_status_result_and_sse_success() -> None:
    client = _client(FakeAutonomousAgentService())

    create_response = client.post("/api/v1/agent/query-jobs", json=_query_payload())

    assert create_response.status_code == 202
    created = create_response.json()
    job_id = created["job_id"]
    assert created["status"] == "queued"
    assert created["links"] == {
        "status": f"/api/v1/agent/query-jobs/{job_id}",
        "result": f"/api/v1/agent/query-jobs/{job_id}/result",
        "events": f"/api/v1/agent/query-jobs/{job_id}/events",
    }

    status_response = client.get(f"/api/v1/agent/query-jobs/{job_id}")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["status"] == "succeeded"
    assert status_payload["progress"] == 100

    result_response = client.get(f"/api/v1/agent/query-jobs/{job_id}/result")
    assert result_response.status_code == 200
    result = result_response.json()
    assert result["route"] == "symbol_comparison"
    assert result["symbols"] == ["AAPL", "MSFT"]
    assert result["result"] == {"route": "symbol_comparison"}

    with client.stream("GET", f"/api/v1/agent/query-jobs/{job_id}/events") as sse_response:
        assert sse_response.status_code == 200
        body = sse_response.read().decode()
    assert "event: status" in body
    assert "event: result" in body
    assert '"status":"succeeded"' in body


def test_agent_query_job_failed_result_returns_error_response() -> None:
    client = _client(FailingAutonomousAgentService())

    create_response = client.post("/api/v1/agent/query-jobs", json=_query_payload())

    assert create_response.status_code == 202
    job_id = create_response.json()["job_id"]

    status_response = client.get(f"/api/v1/agent/query-jobs/{job_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "failed"

    result_response = client.get(f"/api/v1/agent/query-jobs/{job_id}/result")
    assert result_response.status_code == 500
    payload = result_response.json()
    assert payload["status"] == "ERROR"
    assert payload["error_code"] == "INTERNAL_ERROR"
    assert payload["message"] == "agent service failed"


def test_agent_query_job_result_returns_202_while_queued() -> None:
    job_id = "queued-agent-job"
    now = _now_iso()
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "request_id": "UNKNOWN",
            "status": "queued",
            "progress": 0,
            "progress_stage": "queued",
            "run_id": None,
            "error": None,
            "result": None,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    client = _client(FakeAutonomousAgentService())

    response = client.get(f"/api/v1/agent/query-jobs/{job_id}/result")

    assert response.status_code == 202
    assert response.json()["status"] == "queued"


def test_agent_query_job_unknown_returns_404() -> None:
    client = _client(FakeAutonomousAgentService())

    status_response = client.get("/api/v1/agent/query-jobs/missing")
    result_response = client.get("/api/v1/agent/query-jobs/missing/result")
    with client.stream("GET", "/api/v1/agent/query-jobs/missing/events") as sse_response:
        body = sse_response.read().decode()

    assert status_response.status_code == 404
    assert result_response.status_code == 404
    assert result_response.json()["error_code"] == "JOB_NOT_FOUND"
    assert "event: error" in body
    assert "JOB_NOT_FOUND" in body
