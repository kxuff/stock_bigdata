import json
from pathlib import Path
from typing import Any

from app.config import AgentSettings
from app.schemas.decision import SingleSymbolDecision
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle
from app.services import crew_runner
from app.services.crew_runner import HierarchicalCrewRunner


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def load_sample(name: str) -> dict:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


class FakeProcess:
    hierarchical = "hierarchical"


class FakeAgent:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.role = kwargs.get("role")
        self.tools = kwargs.get("tools", [])
        self.allow_delegation = kwargs.get("allow_delegation", False)


class FakeTask:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.agent = kwargs.get("agent")


class FakeCrew:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.agents = kwargs["agents"]
        self.tasks = kwargs["tasks"]
        self.manager_agent = kwargs["manager_agent"]
        self.process = kwargs["process"]
        self.inputs: dict[str, Any] | None = None

    def kickoff(self, inputs: dict[str, Any]) -> str:
        self.inputs = inputs
        return json.dumps(load_sample("normal_final_decision.json"))


def test_hierarchical_crew_uses_custom_manager_and_specialist_tools(monkeypatch) -> None:
    monkeypatch.setattr(crew_runner, "Agent", FakeAgent)
    monkeypatch.setattr(crew_runner, "Task", FakeTask)
    monkeypatch.setattr(crew_runner, "Crew", FakeCrew)
    monkeypatch.setattr(crew_runner, "Process", FakeProcess)

    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("normal_tool_results.json"))
    runner = HierarchicalCrewRunner(
        settings=AgentSettings(),
        llm_factory=lambda settings: object(),
    )

    decision = runner.run(request, bundle)
    artifacts = runner.last_artifacts

    assert isinstance(decision, SingleSymbolDecision)
    assert artifacts is not None
    assert artifacts.crew.process == FakeProcess.hierarchical
    assert artifacts.manager_agent not in artifacts.specialist_agents
    assert artifacts.manager_agent.allow_delegation is True
    assert [agent.role for agent in artifacts.specialist_agents] == [
        "Market Data Analyst",
        "Financial Sentiment Analyst",
        "Valuation Analyst",
        "Risk Analyst",
    ]
    assert [len(agent.tools) for agent in artifacts.specialist_agents] == [2, 1, 1, 2]
    assert len(artifacts.tasks) == 5
    assert artifacts.crew.inputs is not None
    assert artifacts.crew.inputs["request"]["request_id"] == request.request_id


def test_mocked_upstream_tool_returns_configured_bundle_field() -> None:
    bundle = ToolResultBundle.model_validate(load_sample("normal_tool_results.json"))
    tools = crew_runner.build_mocked_upstream_tools(bundle)

    market_payload = json.loads(tools["market_features"]._run("AAPL"))

    assert market_payload["tool"] == "MarketFeatureTool"
    assert market_payload["data"]["AAPL"]["latest_price"] == 276.84
