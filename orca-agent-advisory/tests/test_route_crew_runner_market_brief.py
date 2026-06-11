from app.infrastructure.crewai.route_crew_runner import RouteCrewRunner, _market_brief_limit
from app.schemas.agent import AgentQueryRequest
from app.schemas.enums import AgentRoute
from app.schemas.route_agent_outputs import RouteAgentResponseOutput


class FakeMarketScreenProvider:
    def __init__(self) -> None:
        self.last_limit: int | None = None

    def screen_latest(self, limit: int) -> list[dict]:
        self.last_limit = limit
        return [
            {
                "Symbol": f"S{i}",
                "Datetime": "2026-06-10",
                "final_score": 100 - i,
                "latest_price": 10 + i,
                "sentiment_label": "positive" if i == 0 else None,
                "sentiment_score": 0.72 if i == 0 else None,
                "top_drivers": ["Strong product headline"] if i == 0 else [],
            }
            for i in range(limit)
        ]


def test_market_brief_limit_uses_user_requested_count() -> None:
    assert _market_brief_limit("Show me 10 stocks to watch right now") == 10
    assert _market_brief_limit("show me ten names to watch") == 10
    assert _market_brief_limit("show me 99 stocks") == 20
    assert _market_brief_limit("show me stocks to watch") == 5


def test_market_brief_grounding_repairs_hallucinated_message(monkeypatch) -> None:
    import litellm

    def fake_completion(**kwargs):
        return {"choices": [{"message": {"content": "S0 leads the grounded market brief after repair."}}]}

    monkeypatch.setattr(litellm, "completion", fake_completion)
    provider = FakeMarketScreenProvider()
    runner = RouteCrewRunner(market_screen_provider=provider)
    payload = RouteAgentResponseOutput(
        message="NVDA leads. Risk: data stale as of 2025-02-14.",
        result_type="market_brief",
        result={"leaders": ["NVDA", "SMCI"]},
    )

    grounded = runner._ground_market_brief_payload(
        payload,
        AgentQueryRequest(message="Show me 10 stocks to watch right now"),
    )

    assert provider.last_limit == 10
    assert grounded.message == "S0 leads the grounded market brief after repair."


def test_market_brief_grounding_keeps_valid_agent_message_and_replaces_data() -> None:
    provider = FakeMarketScreenProvider()
    runner = RouteCrewRunner(market_screen_provider=provider)
    payload = RouteAgentResponseOutput(
        message="S0 leads the current market brief with positive news and S1 follows.",
        result_type="market_brief",
        result={"leaders": []},
    )

    grounded = runner._ground_market_brief_payload(
        payload,
        AgentQueryRequest(message="Show me 10 stocks to watch right now"),
    )

    assert provider.last_limit == 10
    assert grounded.result_type == "market_brief"
    assert len(grounded.result["leaders"]) == 10
    assert grounded.result["leaders"][0]["Symbol"] == "S0"
    assert grounded.message == "S0 leads the current market brief with positive news and S1 follows."


def test_top_stocks_grounding_uses_stocks_key_and_default_ten() -> None:
    provider = FakeMarketScreenProvider()
    runner = RouteCrewRunner(market_screen_provider=provider)
    payload = RouteAgentResponseOutput(
        message="Here are the top stocks.",
        result_type="top_stocks",
        result={"stocks": []},
    )

    grounded = runner._ground_market_screen_payload(
        payload,
        AgentQueryRequest(message="Show me top stocks"),
        AgentRoute.TOP_STOCKS,
    )

    assert provider.last_limit == 10
    assert grounded.result_type == "top_stocks"
    assert len(grounded.result["stocks"]) == 10
