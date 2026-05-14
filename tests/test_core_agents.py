import json
from pathlib import Path

from app.agents.data_agent import analyze_market_data, create_market_data_agent
from app.agents.risk_agent import analyze_risk, create_risk_agent
from app.agents.sentiment_agent import analyze_sentiment, create_sentiment_agent
from app.agents.valuation_agent import analyze_valuation, create_valuation_agent
from app.crews.config_loader import load_agents_config, load_tasks_config
from app.schemas.enums import AgentStatus, RiskLabel, SentimentLabel, ToolStatus, ValuationLabel
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle
from app.services.crew_runner import build_mocked_upstream_tools
from app.tasks.data_tasks import create_market_data_task
from app.tasks.risk_tasks import create_risk_task
from app.tasks.sentiment_tasks import create_sentiment_task
from app.tasks.valuation_tasks import create_valuation_task


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


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


def test_agent_factories_wire_yaml_config_and_read_only_tools() -> None:
    request = AdvisoryDecisionRequest.model_validate(load_sample("normal_request.json"))
    bundle = ToolResultBundle.model_validate(load_sample("normal_tool_results.json"))
    tools = build_mocked_upstream_tools(bundle)
    llm = "openai/gpt-4o-mini"

    market_agent = create_market_data_agent(
        llm=llm,
        tools=[tools["market_features"], tools["ml_predictions"]],
    )
    sentiment_agent = create_sentiment_agent(llm=llm, tools=[tools["sentiment_snapshot"]])
    valuation_agent = create_valuation_agent(llm=llm, tools=[tools["valuation_snapshot"]])
    risk_agent = create_risk_agent(
        llm=llm,
        tools=[tools["risk_snapshot"], tools["portfolio_snapshot"]],
    )

    assert market_agent.role == "Market Data Analyst"
    assert sentiment_agent.role == "Financial Sentiment Analyst"
    assert valuation_agent.role == "Valuation Analyst"
    assert risk_agent.role == "Risk Analyst"
    assert [len(agent.tools) for agent in [market_agent, sentiment_agent, valuation_agent, risk_agent]] == [
        2,
        1,
        1,
        2,
    ]
    assert request.symbols == ["AAPL"]


def test_task_factories_use_yaml_prompt_json_contracts() -> None:
    bundle = ToolResultBundle.model_validate(load_sample("normal_tool_results.json"))
    tools = build_mocked_upstream_tools(bundle)
    llm = "openai/gpt-4o-mini"
    market_agent = create_market_data_agent(
        llm=llm,
        tools=[tools["market_features"], tools["ml_predictions"]],
    )
    sentiment_agent = create_sentiment_agent(llm=llm, tools=[tools["sentiment_snapshot"]])
    valuation_agent = create_valuation_agent(llm=llm, tools=[tools["valuation_snapshot"]])
    risk_agent = create_risk_agent(llm=llm, tools=[tools["risk_snapshot"]])

    market_task = create_market_data_task(market_agent)
    sentiment_task = create_sentiment_task(sentiment_agent)
    valuation_task = create_valuation_task(valuation_agent)
    risk_task = create_risk_task(risk_agent)

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
