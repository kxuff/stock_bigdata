import json
from dataclasses import dataclass, field
from typing import Any, Callable

from app.agents.manager_agent import create_manager_agent
from app.config import AgentSettings, load_settings
from app.crews.advisory_crew import AdvisorySpecialistCrew
from app.llm.llm_factory import create_deepseek_llm
from app.schemas.decision import SingleSymbolDecision
from app.schemas.manager_outputs import ManagerSynthesisOutput
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle
from app.tasks.manager_tasks import create_manager_synthesis_task
from app.validators.manager_synthesis_parser import parse_manager_synthesis_output
from app.validators.output_repair import parse_model_output

try:
    from crewai import Crew, Process
    from crewai.tools import BaseTool
except ModuleNotFoundError:
    Crew = None
    Process = None
    BaseTool = object


@dataclass
class CrewRunArtifacts:
    manager_agent: Any
    specialist_agents: list[Any]
    tasks: list[Any]
    crew: Any


@dataclass
class HierarchicalCrewRunner:
    settings: AgentSettings = field(default_factory=load_settings)
    llm_factory: Callable[[AgentSettings], Any] = create_deepseek_llm
    verbose: bool = False
    last_artifacts: CrewRunArtifacts | None = None

    def run(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
        *,
        output_model: type[SingleSymbolDecision] = SingleSymbolDecision,
    ) -> SingleSymbolDecision:
        artifacts = self.build_crew(request, tool_results)
        raw_result = artifacts.crew.kickoff(
            inputs={
                "request": request.model_dump(mode="json"),
                "request_json": request.model_dump_json(),
            }
        )
        return parse_model_output(raw_result, output_model)

    def run_manager_synthesis(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
    ) -> ManagerSynthesisOutput:
        artifacts = self.build_crew(request, tool_results)
        raw_result = artifacts.crew.kickoff(
            inputs={
                "request": request.model_dump(mode="json"),
                "request_json": request.model_dump_json(),
                "symbols": ", ".join(request.symbols),
                "decision_mode": request.decision_mode.value,
            }
        )

        pydantic_output = getattr(raw_result, "pydantic", None)
        if isinstance(pydantic_output, ManagerSynthesisOutput):
            return pydantic_output
        return parse_manager_synthesis_output(raw_result, request)

    def build_crew(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
    ) -> CrewRunArtifacts:
        _require_crewai()
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


def _require_crewai() -> None:
    if Crew is None or Process is None:
        raise RuntimeError("CrewAI is required for the hierarchical crew runner")
