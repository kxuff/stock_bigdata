from typing import Any

from app.infrastructure.crewai.agents.data_agent import create_market_data_agent
from app.infrastructure.crewai.agents.risk_agent import create_risk_agent
from app.infrastructure.crewai.agents.sentiment_agent import create_sentiment_agent
from app.infrastructure.crewai.agents.valuation_agent import create_valuation_agent
from app.infrastructure.crewai.tasks.data_tasks import create_market_data_task
from app.infrastructure.crewai.tasks.risk_tasks import create_risk_task
from app.infrastructure.crewai.tasks.sentiment_tasks import create_sentiment_task
from app.infrastructure.crewai.tasks.valuation_tasks import create_valuation_task

try:
    from crewai import Crew, Process
    from crewai.project import CrewBase, agent, crew, task
except ModuleNotFoundError:
    Crew = None
    Process = None

    def CrewBase(cls: type[Any]) -> type[Any]:
        return cls

    def agent(func: Any) -> Any:
        return func

    def task(func: Any) -> Any:
        return func

    def crew(func: Any) -> Any:
        return func


@CrewBase
class AdvisorySpecialistCrew:
    """CrewAI-native specialist crew using YAML-backed agent and task config."""

    agents_config = "../config/agents.yaml"
    tasks_config = "../config/tasks.yaml"

    def __init__(
        self,
        *,
        llm: Any,
        tools: dict[str, Any],
        manager_agent: Any,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self._manager_agent = manager_agent
        self.verbose = verbose

    @agent
    def manager_agent(self) -> Any:
        return self._manager_agent

    @agent
    def market_data_agent(self) -> Any:
        return create_market_data_agent(
            llm=self.llm,
            tools=[self.tools["market_features"], self.tools["ml_predictions"]],
            verbose=self.verbose,
        )

    @agent
    def sentiment_agent(self) -> Any:
        return create_sentiment_agent(
            llm=self.llm,
            tools=[self.tools["sentiment_snapshot"]],
            verbose=self.verbose,
        )

    @agent
    def valuation_agent(self) -> Any:
        return create_valuation_agent(
            llm=self.llm,
            tools=[self.tools["valuation_snapshot"]],
            verbose=self.verbose,
        )

    @agent
    def risk_agent(self) -> Any:
        return create_risk_agent(
            llm=self.llm,
            tools=[self.tools["risk_snapshot"], self.tools["portfolio_snapshot"]],
            verbose=self.verbose,
        )

    @task
    def market_data_task(self) -> Any:
        return create_market_data_task(self.market_data_agent())

    @task
    def sentiment_task(self) -> Any:
        return create_sentiment_task(self.sentiment_agent())

    @task
    def valuation_task(self) -> Any:
        return create_valuation_task(self.valuation_agent())

    @task
    def risk_task(self) -> Any:
        return create_risk_task(self.risk_agent())

    @crew
    def crew(self) -> Any:
        _require_crewai()
        return Crew(
            agents=self.specialist_agents(),
            tasks=self.specialist_tasks(),
            manager_agent=self._manager_agent,
            process=Process.hierarchical,
            verbose=self.verbose,
        )

    def specialist_agents(self) -> list[Any]:
        return [
            self.market_data_agent(),
            self.sentiment_agent(),
            self.valuation_agent(),
            self.risk_agent(),
        ]

    def specialist_tasks(self) -> list[Any]:
        return [
            self.market_data_task(),
            self.sentiment_task(),
            self.valuation_task(),
            self.risk_task(),
        ]


def _require_crewai() -> None:
    if Crew is None or Process is None:
        raise RuntimeError("CrewAI is required to create AdvisorySpecialistCrew")
