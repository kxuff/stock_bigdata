import json
from pathlib import Path
from typing import Any

from app.application.specialists import analyze_market_data
from app.application.specialists import analyze_risk
from app.application.specialists import analyze_sentiment
from app.application.specialists import analyze_valuation
from app.infrastructure.crewai.agents import data_agent, risk_agent, sentiment_agent, valuation_agent
from app.infrastructure.crewai.agents.data_agent import create_market_data_agent
from app.infrastructure.crewai.agents.risk_agent import create_risk_agent
from app.infrastructure.crewai.agents.sentiment_agent import create_sentiment_agent
from app.infrastructure.crewai.agents.valuation_agent import create_valuation_agent
from app.infrastructure.crewai.config_loader import load_agents_config, load_tasks_config
from app.infrastructure.crewai.crew_runner import build_mocked_upstream_tools
from app.infrastructure.crewai.tasks import data_tasks, risk_tasks, sentiment_tasks, valuation_tasks
from app.infrastructure.crewai.tasks.data_tasks import create_market_data_task
from app.infrastructure.crewai.tasks.risk_tasks import create_risk_task
from app.infrastructure.crewai.tasks.sentiment_tasks import create_sentiment_task
from app.infrastructure.crewai.tasks.valuation_tasks import create_valuation_task
from app.schemas.enums import AgentStatus, RiskLabel, SentimentLabel, ToolStatus, ValuationLabel
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


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
    def __init__(self, *, config: dict[str, Any], agent: Any | None = None, **_: Any) -> None:
        self.config = config
        self.agent = agent
        self.output_pydantic = None
        self.expected_output = config.get("expected_output")


def load_sample(name: str) -> dict:
    return json.loads((SAMPLES_DIR / name).read_text(encoding="utf-8"))


def test_crewai_yaml_configs_define_specialists_and_prompt_boundaries() -> None:
    agents_config = load_agents_config()
    tasks_config = load_tasks_config()

    assert set(agents_config) == {
        "market_data_agent",
        "sentiment_agent",
        "valuation_agent",
        "risk_agent",
        "manager_agent",
    }
    assert tasks_config["market_data_task"]["agent"] == "market_data_agent"
    assert tasks_config["manager_synthesis_task"]["agent"] == "manager_agent"
    assert "untrusted evidence" in tasks_config["sentiment_task"]["description"]
    assert "Do not infer valuation metrics" in tasks_config["valuation_task"]["description"]
    assert "decision_rationale[].stance must be one of" in tasks_config["manager_synthesis_task"]["description"]


def test_agent_factories_wire_yaml_config_and_read_only_tools(monkeypatch) -> None:
    monkeypatch.setattr(data_agent, "Agent", FakeAgent)
    monkeypatch.setattr(sentiment_agent, "Agent", FakeAgent)
    monkeypatch.setattr(valuation_agent, "Agent", FakeAgent)
    monkeypatch.setattr(risk_agent, "Agent", FakeAgent)

    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("normal_tool_results.json"))
    tools = build_mocked_upstream_tools(bundle)
    llm = "deepseek/deepseek-v4-flash"

    market_agent = create_market_data_agent(
        llm=llm,
        tools=[tools["market_features"], tools["ml_predictions"]],
    )
    sentiment_agent_obj = create_sentiment_agent(llm=llm, tools=[tools["sentiment_snapshot"]])
    valuation_agent_obj = create_valuation_agent(llm=llm, tools=[tools["valuation_snapshot"]])
    risk_agent_obj = create_risk_agent(
        llm=llm,
        tools=[tools["risk_snapshot"], tools["portfolio_snapshot"]],
    )

    assert market_agent.role == "Market Data Analyst"
    assert sentiment_agent_obj.role == "Financial Sentiment Analyst"
    assert valuation_agent_obj.role == "Valuation Analyst"
    assert risk_agent_obj.role == "Risk Analyst"
    assert [len(agent.tools) for agent in [market_agent, sentiment_agent_obj, valuation_agent_obj, risk_agent_obj]] == [
        2,
        1,
        1,
        2,
    ]
    assert request.symbols == ["AAPL"]


def test_task_factories_use_yaml_prompt_json_contracts(monkeypatch) -> None:
    monkeypatch.setattr(data_agent, "Agent", FakeAgent)
    monkeypatch.setattr(sentiment_agent, "Agent", FakeAgent)
    monkeypatch.setattr(valuation_agent, "Agent", FakeAgent)
    monkeypatch.setattr(risk_agent, "Agent", FakeAgent)
    monkeypatch.setattr(data_tasks, "Task", FakeTask)
    monkeypatch.setattr(sentiment_tasks, "Task", FakeTask)
    monkeypatch.setattr(valuation_tasks, "Task", FakeTask)
    monkeypatch.setattr(risk_tasks, "Task", FakeTask)

    bundle = ToolResultBundle.model_validate(load_sample("normal_tool_results.json"))
    tools = build_mocked_upstream_tools(bundle)
    llm = "deepseek/deepseek-v4-flash"
    market_agent = create_market_data_agent(
        llm=llm,
        tools=[tools["market_features"], tools["ml_predictions"]],
    )
    sentiment_agent_obj = create_sentiment_agent(llm=llm, tools=[tools["sentiment_snapshot"]])
    valuation_agent_obj = create_valuation_agent(llm=llm, tools=[tools["valuation_snapshot"]])
    risk_agent_obj = create_risk_agent(llm=llm, tools=[tools["risk_snapshot"]])

    market_task = create_market_data_task(market_agent)
    sentiment_task = create_sentiment_task(sentiment_agent_obj)
    valuation_task = create_valuation_task(valuation_agent_obj)
    risk_task = create_risk_task(risk_agent_obj)

    assert market_task.output_pydantic is None
    assert sentiment_task.output_pydantic is None
    assert valuation_task.output_pydantic is None
    assert risk_task.output_pydantic is None
    assert "Return only valid JSON" in market_task.expected_output
    assert "Return only valid JSON" in sentiment_task.expected_output
    assert "Return only valid JSON" in valuation_task.expected_output
    assert "Return only valid JSON" in risk_task.expected_output


def test_market_data_agent_output_uses_tool_results_without_inventing_metrics() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("normal_tool_results.json"))

    output = analyze_market_data(request, bundle)

    assert output.status == AgentStatus.SUCCESS
    assert output.market_signals[0].symbol == "AAPL"
    assert output.market_signals[0].confidence == 0.65
    assert output.ml_signal_available is True
    assert any("ml_probability_up=0.64" in driver for driver in output.market_signals[0].drivers)


def test_sentiment_agent_skips_when_optional_tool_result_missing() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("missing_sentiment_tool_results.json"))

    output = analyze_sentiment(request, bundle)

    assert output.status == AgentStatus.SKIPPED
    assert output.sentiment_label == SentimentLabel.UNAVAILABLE
    assert output.missing_fields == ["sentiment_snapshot"]
    assert output.limitations == ["SENTIMENT_CONTEXT_UNAVAILABLE"]


def test_valuation_agent_skips_when_optional_tool_result_missing() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("missing_valuation_tool_results.json"))

    output = analyze_valuation(request, bundle)

    assert output.status == AgentStatus.SKIPPED
    assert output.valuation_label == ValuationLabel.UNKNOWN
    assert output.missing_fields == ["valuation_snapshot"]
    assert output.limitations == ["VALUATION_CONTEXT_UNAVAILABLE"]


def test_valuation_agent_degrades_low_quality_metadata() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    payload = load_sample("normal_tool_results.json")
    payload["valuation_snapshot"]["data"]["AAPL"].update(
        {
            "valuation_quality": "LOW",
            "valuation_method": "relative_pe",
            "sector_sample_count": 3,
        }
    )
    bundle = ToolResultBundle.model_validate(payload)

    output = analyze_valuation(request, bundle)

    assert output.status == AgentStatus.DEGRADED
    assert output.confidence == 0.45
    assert "VALUATION_QUALITY_LOW" in output.limitations
    assert "AAPL valuation_method=relative_pe" in output.valuation_drivers
    assert "AAPL valuation_quality=LOW" in output.valuation_drivers
    assert "AAPL sector_sample_count=3" in output.valuation_drivers


def test_risk_agent_returns_risk_label_factors_and_confidence_cap() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("portfolio_allocation_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("portfolio_allocation_tool_results.json"))

    output = analyze_risk(request, bundle)

    assert output.status == AgentStatus.SUCCESS
    assert output.risk_label == RiskLabel.MEDIUM
    assert output.confidence_cap == 0.78
    assert any("single asset weight exceeds user constraint: AAPL" in factor for factor in output.risk_factors)


def test_risk_agent_degrades_when_portfolio_tool_is_not_successful_in_portfolio_mode() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("portfolio_allocation_request.json"))
    payload = load_sample("portfolio_allocation_tool_results.json")
    payload["portfolio_snapshot"]["status"] = ToolStatus.UNAVAILABLE
    payload["portfolio_snapshot"]["error_message"] = "portfolio service unavailable"
    bundle = ToolResultBundle.model_validate(payload)

    output = analyze_risk(request, bundle)

    assert output.status == AgentStatus.DEGRADED
    assert "portfolio_snapshot" in output.missing_fields
    assert "PORTFOLIO_CONTEXT_UNAVAILABLE" in output.limitations
