from typing import Any, cast

from crewai import Crew, Process
from crewai.project import CrewBase as CrewBaseDecorator
from crewai.project import agent as crew_agent_decorator
from crewai.project import crew as crew_decorator
from crewai.project import task as crew_task_decorator

from app.infrastructure.crewai.agents.data_agent import create_market_data_agent
from app.infrastructure.crewai.agents.risk_agent import create_risk_agent
from app.infrastructure.crewai.agents.sentiment_agent import create_sentiment_agent
from app.infrastructure.crewai.agents.valuation_agent import create_valuation_agent
from app.infrastructure.crewai.tasks.data_tasks import create_market_data_task
from app.infrastructure.crewai.tasks.manager_tasks import create_manager_synthesis_task
from app.infrastructure.crewai.tasks.risk_tasks import create_risk_task
from app.infrastructure.crewai.tasks.sentiment_tasks import create_sentiment_task
from app.infrastructure.crewai.tasks.valuation_tasks import create_valuation_task

CrewBase = cast(Any, CrewBaseDecorator)
agent = cast(Any, crew_agent_decorator)
crew = cast(Any, crew_decorator)
task = cast(Any, crew_task_decorator)


@CrewBase
class AdvisoryHierarchicalCrew:
    """Hierarchical crew: manager agent orchestrates 4 specialist agents.

    Manager is NOT in agents= list — it is specified via manager_agent= on Crew,
    which is the correct pattern for Process.hierarchical. The manager receives
    all specialist task outputs and produces the final synthesis automatically.
    """

    def __init__(
        self,
        *,
        llm: Any,
        tools: dict[str, Any],
        manager_agent: Any,
        verbose: bool = False,
        tracing: bool = False,
        share_crew: bool = False,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self._manager_agent = manager_agent
        self.verbose = verbose
        self.tracing = tracing
        self.share_crew = share_crew

    # --- Specialist agents (manager is NOT decorated here) ---

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

    # --- Specialist tasks with explicit agent assignments ---

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

    @task
    def manager_synthesis_task(self) -> Any:
        return create_manager_synthesis_task(
            specialist_tasks=[
                self.market_data_task(),
                self.sentiment_task(),
                self.valuation_task(),
                self.risk_task(),
            ],
        )

    # --- Hierarchical crew: manager_agent= keeps manager out of agents list ---

    @crew
    def crew(self) -> Any:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_agent=self._manager_agent,
            verbose=self.verbose,
            tracing=self.tracing,
            share_crew=self.share_crew,
        )
