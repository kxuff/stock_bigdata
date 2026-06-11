from app.infrastructure.crewai.route_crew_runner import RouteCrewRunner, _market_brief_limit
from app.schemas.agent import AgentQueryRequest
from app.schemas.route_agent_outputs import RouteAgentResponseOutput


class FakeMarketScreenProvider:
    def __init__(self) -> None:
        self.last_limit: int | None = None

    def screen_latest(self, limit: int) -> list[dict]:
        self.last_limit = limit
        return [
            {"Symbol": f"S{i}", "Datetime": "2026-06-10", "final_score": 100 - i, "latest_price": 10 + i}
            for i in range(limit)
        ]


def test_market_brief_limit_uses_user_requested_count() -> None:
    assert _market_brief_limit("Show me 10 stocks to watch right now") == 10
    assert _market_brief_limit("show me ten names to watch") == 10
    assert _market_brief_limit("show me 99 stocks") == 20
    assert _market_brief_limit("show me stocks to watch") == 5


def test_market_brief_grounding_replaces_hallucinated_leaders_and_date() -> None:
    provider = FakeMarketScreenProvider()
    runner = RouteCrewRunner(market_screen_provider=provider)
    payload = RouteAgentResponseOutput(
        message="NVDA leads. Risk: data stale as of 2025-02-14. Not financial advice.",
        result_type="market_brief",
        result={"leaders": ["NVDA", "SMCI"]},
    )

    grounded = runner._ground_market_brief_payload(
        payload,
        AgentQueryRequest(message="Show me 10 stocks to watch right now"),
    )

    assert provider.last_limit == 10
    assert grounded.result_type == "market_brief"
    assert len(grounded.result["leaders"]) == 10
    assert grounded.result["leaders"][0]["Symbol"] == "S0"
    assert "2025-02-14" not in grounded.message
    assert "2026-06-10" in grounded.message
    assert grounded.message.endswith("Not financial advice.")
