import json
from pathlib import Path
from typing import Any

from app.config import AgentSettings
from app.infrastructure.crewai.agents import data_agent, manager_agent, risk_agent, sentiment_agent, valuation_agent
from app.infrastructure.crewai import crew_runner
from app.infrastructure.crewai.crew_runner import HierarchicalCrewRunner
from app.infrastructure.crewai.tasks import data_tasks, manager_tasks, risk_tasks, sentiment_tasks, valuation_tasks
from app.schemas.decision import SingleSymbolDecision
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def load_sample(name: str) -> dict:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


class FakeProcess:
    hierarchical = "hierarchical"


class FakeAgent:
    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        llm: Any,
        tools: list[Any] | None = None,
        verbose: bool = False,
        allow_delegation: bool = False,
        max_iter: int | None = None,
        max_execution_time: int | None = None,
    ) -> None:
        self.config = config or {}
        self.llm = llm
        self.role = self.config.get("role")
        self.tools = tools or []
        self.verbose = verbose
        self.allow_delegation = allow_delegation
        self.max_iter = max_iter
        self.max_execution_time = max_execution_time


class FakeTask:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        agent: Any | None = None,
        context: list[Any] | None = None,
        output_pydantic: Any | None = None,
        expected_output: str | None = None,
    ) -> None:
        self.name = config.get("name")
        self.config = config
        self.agent = agent
        self.context = context or []
        self.output_pydantic = output_pydantic
        self.expected_output = expected_output or config.get("expected_output")


class FakeCrew:
    def __init__(
        self,
        *,
        agents: list[Any],
        tasks: list[Any],
        manager_agent: Any,
        process: Any,
        verbose: bool = False,
        tracing: bool = True,
        share_crew: bool = False,
    ) -> None:
        self.agents = agents
        self.tasks = tasks
        self.manager_agent = manager_agent
        self.process = process
        self.verbose = verbose
        self.tracing = tracing
        self.share_crew = share_crew
        self.inputs: dict[str, Any] | None = None

    def kickoff(self, inputs: dict[str, Any]) -> str:
        self.inputs = inputs
        return json.dumps(load_sample("normal_final_decision.json"))


def test_hierarchical_crew_uses_custom_manager_and_specialist_tools(monkeypatch) -> None:
    monkeypatch.setattr(manager_agent, "Agent", FakeAgent)
    monkeypatch.setattr(data_agent, "Agent", FakeAgent)
    monkeypatch.setattr(sentiment_agent, "Agent", FakeAgent)
    monkeypatch.setattr(valuation_agent, "Agent", FakeAgent)
    monkeypatch.setattr(risk_agent, "Agent", FakeAgent)
    monkeypatch.setattr(data_tasks, "Task", FakeTask)
    monkeypatch.setattr(sentiment_tasks, "Task", FakeTask)
    monkeypatch.setattr(valuation_tasks, "Task", FakeTask)
    monkeypatch.setattr(risk_tasks, "Task", FakeTask)
    monkeypatch.setattr(manager_tasks, "Task", FakeTask)
    monkeypatch.setattr(crew_runner, "Crew", FakeCrew)
    monkeypatch.setattr(crew_runner, "Process", FakeProcess)

    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("normal_tool_results.json"))
    runner = HierarchicalCrewRunner(
        settings=AgentSettings(),
        llm_factory=lambda settings: "deepseek/deepseek-v4-flash",
    )

    decision = runner.run(request, bundle)
    artifacts = runner.last_artifacts

    assert isinstance(decision, SingleSymbolDecision)
    assert artifacts is not None
    assert artifacts.crew.process == FakeProcess.hierarchical
    assert artifacts.manager_agent not in artifacts.specialist_agents
    assert artifacts.manager_agent.allow_delegation is True
    assert artifacts.manager_agent.role == "Investment Advisory Manager"
    assert artifacts.crew.tracing is True
    assert artifacts.crew.share_crew is False
    assert [agent.role for agent in artifacts.specialist_agents] == [
        "Market Data Analyst",
        "Financial Sentiment Analyst",
        "Valuation Analyst",
        "Risk Analyst",
    ]
    assert [len(agent.tools) for agent in artifacts.specialist_agents] == [2, 1, 1, 2]
    assert len(artifacts.tasks) == 5
    assert all(getattr(task, "output_pydantic", None) is None for task in artifacts.tasks)
    assert "Return only valid JSON" in artifacts.tasks[-1].expected_output
    assert artifacts.tasks[-1].context == artifacts.tasks[:4]
    assert artifacts.crew.inputs is not None
    assert artifacts.crew.inputs["request"]["request_id"] == request.request_id


def test_mocked_upstream_tool_returns_configured_bundle_field() -> None:
    bundle = ToolResultBundle.model_validate(load_sample("normal_tool_results.json"))
    tools = crew_runner.build_mocked_upstream_tools(bundle)

    market_payload = json.loads(tools["market_features"]._run("AAPL"))

    assert market_payload["tool"] == "MarketFeatureTool"
    assert market_payload["data"]["AAPL"]["latest_price"] == 276.84
