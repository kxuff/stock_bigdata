import json
from dataclasses import dataclass, field
from typing import Any, Callable

from app.application.specialists import analyze_market_data
from app.application.specialists import analyze_risk
from app.application.specialists import analyze_sentiment
from app.application.specialists import analyze_valuation
from app.infrastructure.crewai.agents.manager_agent import create_manager_agent
from app.config import AgentSettings, load_settings
from app.infrastructure.crewai.crews.advisory_crew import AdvisorySpecialistCrew
from app.application.ports.crew_orchestrator import CrewOrchestratedOutputs
from app.infrastructure.llm.llm_factory import create_llm
from app.schemas.agent_outputs import (
    AgentOutputBundle,
    MarketDataAgentOutput,
    RiskAgentOutput,
    SentimentAgentOutput,
    ValuationAgentOutput,
)
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle
from app.infrastructure.crewai.tasks.manager_tasks import create_manager_synthesis_task
from app.validators.output_repair import parse_model_output

from crewai import Crew, Process
from crewai.tools import BaseTool


@dataclass
class CrewRunArtifacts:
    manager_agent: Any
    specialist_agents: list[Any]
    tasks: list[Any]
    crew: Any


@dataclass
class HierarchicalCrewRunner:
    settings: AgentSettings = field(default_factory=load_settings)
    llm_factory: Callable[[AgentSettings], Any] = create_llm
    verbose: bool = False
    last_artifacts: CrewRunArtifacts | None = None

    def run_orchestrated(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
    ) -> CrewOrchestratedOutputs:
        artifacts = self.build_crew(request, tool_results)
        raw_result = artifacts.crew.kickoff(
            inputs={
                "request": request.model_dump(mode="json"),
                "request_json": request.model_dump_json(),
                "symbols": ", ".join(request.symbols),
                "decision_mode": request.decision_mode.value,
            }
        )

        manager_task = artifacts.tasks[-1] if artifacts.tasks else None
        manager_payload = _extract_task_payload(manager_task) or raw_result
        agent_outputs = _parse_specialist_outputs(artifacts.tasks, request, tool_results)
        return CrewOrchestratedOutputs(
            agent_outputs=agent_outputs,
            manager_payload=manager_payload,
        )

    def build_crew(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
    ) -> CrewRunArtifacts:
        llm = self.llm_factory(self.settings)
        tools = build_mocked_upstream_tools(tool_results)

        manager_agent = create_manager_agent(
            llm=llm,
            verbose=self.verbose,
            max_execution_time=self.settings.agent_timeout_seconds,
        )

        specialist_crew = AdvisorySpecialistCrew(
            llm=llm,
            tools=tools,
            manager_agent=manager_agent,
            verbose=self.verbose,
        )
        specialist_agents = specialist_crew.specialist_agents()
        specialist_tasks = specialist_crew.specialist_tasks()
        tasks = specialist_tasks + [_build_manager_task(request, specialist_tasks)]
        crew = Crew(
            agents=specialist_agents,
            tasks=tasks,
            manager_agent=manager_agent,
            process=Process.hierarchical,
            verbose=self.verbose,
            tracing=self.settings.crewai_tracing,
            share_crew=self.settings.crewai_share_crew,
        )

        artifacts = CrewRunArtifacts(
            manager_agent=manager_agent,
            specialist_agents=specialist_agents,
            tasks=tasks,
            crew=crew,
        )
        self.last_artifacts = artifacts
        return artifacts


def build_mocked_upstream_tools(tool_results: ToolResultBundle) -> dict[str, Any]:
    return {
        "market_features": _StaticTool(
            name="MarketFeatureTool",
            description="Read-only mocked market feature snapshot lookup.",
            bundle_field="market_features",
            tool_results=tool_results,
        ),
        "ml_predictions": _StaticTool(
            name="MlPredictionTool",
            description="Read-only mocked machine learning prediction lookup.",
            bundle_field="ml_predictions",
            tool_results=tool_results,
        ),
        "sentiment_snapshot": _StaticTool(
            name="NewsSentimentTool",
            description="Read-only mocked news sentiment snapshot lookup.",
            bundle_field="sentiment_snapshot",
            tool_results=tool_results,
        ),
        "valuation_snapshot": _StaticTool(
            name="FundamentalsTool",
            description="Read-only mocked fundamentals and valuation lookup.",
            bundle_field="valuation_snapshot",
            tool_results=tool_results,
        ),
        "risk_snapshot": _StaticTool(
            name="RiskFeatureTool",
            description="Read-only mocked risk feature lookup.",
            bundle_field="risk_snapshot",
            tool_results=tool_results,
        ),
        "portfolio_snapshot": _StaticTool(
            name="PortfolioTool",
            description="Read-only mocked portfolio snapshot lookup.",
            bundle_field="portfolio_snapshot",
            tool_results=tool_results,
        ),
    }


class _StaticTool(BaseTool):
    name: str
    description: str
    bundle_field: str
    tool_results: ToolResultBundle

    def _run(self, query: str = "") -> str:
        result = getattr(self.tool_results, self.bundle_field)
        if result is None:
            return json.dumps(
                {
                    "tool": self.name,
                    "status": "UNAVAILABLE",
                    "query": query,
                    "error_message": f"{self.bundle_field} was not provided",
                }
            )
        return result.model_dump_json()


def _build_manager_task(
    request: AdvisoryDecisionRequest,
    specialist_tasks: list[Any],
) -> Any:
    return create_manager_synthesis_task(
        request,
        specialist_tasks,
    )


def _extract_task_payload(task: Any | None) -> Any | None:
    if task is None:
        return None

    for attr in ("output", "result", "raw_output", "raw", "response"):
        value = getattr(task, attr, None)
        if value is not None:
            return value
    return None


def _parse_specialist_outputs(
    tasks: list[Any],
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
) -> AgentOutputBundle:
    market_task = tasks[0] if len(tasks) > 0 else None
    sentiment_task = tasks[1] if len(tasks) > 1 else None
    valuation_task = tasks[2] if len(tasks) > 2 else None
    risk_task = tasks[3] if len(tasks) > 3 else None

    market_output = _parse_specialist_payload(
        _extract_task_payload(market_task),
        MarketDataAgentOutput,
        fallback=lambda: analyze_market_data(request, tool_results),
    )
    sentiment_output = _parse_specialist_payload(
        _extract_task_payload(sentiment_task),
        SentimentAgentOutput,
        fallback=lambda: analyze_sentiment(request, tool_results),
    )
    valuation_output = _parse_specialist_payload(
        _extract_task_payload(valuation_task),
        ValuationAgentOutput,
        fallback=lambda: analyze_valuation(request, tool_results),
    )
    risk_output = _parse_specialist_payload(
        _extract_task_payload(risk_task),
        RiskAgentOutput,
        fallback=lambda: analyze_risk(request, tool_results),
    )

    return AgentOutputBundle(
        market_data_agent=market_output,
        sentiment_agent=sentiment_output,
        valuation_agent=valuation_output,
        risk_agent=risk_output,
    )


def _parse_specialist_payload(
    payload: Any | None,
    model_type: type[MarketDataAgentOutput | SentimentAgentOutput | ValuationAgentOutput | RiskAgentOutput],
    *,
    fallback: Callable[[], Any],
) -> Any:
    if payload is None:
        return fallback()

    pydantic_output = getattr(payload, "pydantic", None)
    if isinstance(pydantic_output, model_type):
        return pydantic_output
    if isinstance(payload, model_type):
        return payload

    if isinstance(payload, dict):
        return model_type.model_validate(payload)
    return parse_model_output(payload, model_type)
