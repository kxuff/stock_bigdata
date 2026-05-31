from app.application.ports.crew_orchestrator import CrewOrchestrator
from app.application.ports.output_store import DecisionOutputStore as DecisionOutputStorePort
from app.application.ports.tool_result_provider import ToolResultProvider
from app.application.use_cases.advisory_decision_service import AdvisoryDecisionService
from app.config import AgentSettings, load_settings
from app.infrastructure.bigdata.bigdata_ml_provider import BigdataMlToolResultProvider
from app.infrastructure.crewai.crew_runner import HierarchicalCrewRunner
from app.infrastructure.storage.output_store import DecisionOutputStore


def build_tool_result_provider(settings: AgentSettings) -> ToolResultProvider:
    return BigdataMlToolResultProvider()


def build_crew_orchestrator(settings: AgentSettings) -> CrewOrchestrator:
    return HierarchicalCrewRunner(settings=settings, verbose=settings.crewai_verbose)


def build_output_store(settings: AgentSettings) -> DecisionOutputStorePort:
    return DecisionOutputStore(settings.advisory_output_dir)


def build_decision_service(settings: AgentSettings | None = None) -> AdvisoryDecisionService:
    resolved_settings = settings or load_settings()
    return AdvisoryDecisionService(
        settings=resolved_settings,
        crew_runner=build_crew_orchestrator(resolved_settings),
        output_store=build_output_store(resolved_settings),
    )
