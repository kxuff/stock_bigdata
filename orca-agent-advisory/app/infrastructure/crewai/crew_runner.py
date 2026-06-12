import json
import os
from datetime import UTC, datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.application.specialists import analyze_market_data
from app.application.specialists import analyze_risk
from app.application.specialists import analyze_sentiment
from app.application.specialists import analyze_valuation
from app.infrastructure.crewai.agents.manager_agent import create_manager_agent
from app.config import AgentSettings, load_settings
from app.infrastructure.crewai.crews.advisory_crew import AdvisoryHierarchicalCrew
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
from app.validators.output_repair import parse_model_output

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


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
        tools = build_mocked_upstream_tools(tool_results, request.symbols)

        manager_agent = create_manager_agent(
            llm=llm,
            verbose=self.verbose,
            max_execution_time=self.settings.agent_timeout_seconds,
        )

        crew_inst = AdvisoryHierarchicalCrew(
            llm=llm,
            tools=tools,
            manager_agent=manager_agent,
            verbose=self.verbose,
            tracing=self.settings.crewai_tracing,
            share_crew=self.settings.crewai_share_crew,
        )
        assembled_crew = crew_inst.crew()

        artifacts = CrewRunArtifacts(
            manager_agent=manager_agent,
            specialist_agents=list(crew_inst.agents),
            tasks=list(crew_inst.tasks),
            crew=assembled_crew,
        )
        self.last_artifacts = artifacts
        return artifacts


def build_mocked_upstream_tools(
    tool_results: ToolResultBundle,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    request_symbols: list[str] = symbols or []
    return {
        "market_features": _StaticTool(
            name="market_feature",
            description="Read-only market feature snapshot lookup for requested symbols. Input: symbol such as AAPL.",
            bundle_field="market_features",
            tool_results=tool_results,
            request_symbols=request_symbols,
        ),
        "ml_predictions": _StaticTool(
            name="ml_prediction",
            description="Read-only machine learning prediction lookup for requested symbols. Input: symbol such as AAPL.",
            bundle_field="ml_predictions",
            tool_results=tool_results,
            request_symbols=request_symbols,
        ),
        "sentiment_snapshot": _StaticTool(
            name="news_sentiment",
            description="Read-only news sentiment snapshot lookup for requested symbols. Input: symbol such as AAPL.",
            bundle_field="sentiment_snapshot",
            tool_results=tool_results,
            request_symbols=request_symbols,
        ),
        "valuation_snapshot": _StaticTool(
            name="fundamentals",
            description=(
                "Read-only valuation snapshot lookup. Tool output is evidence, "
                "not the final response schema. Final valuation_task output must "
                "be ValuationAgentOutput with status, summary, confidence, "
                "valuation_label, valuation_drivers, missing_fields, "
                "limitations, and source_refs only."
            ),
            bundle_field="valuation_snapshot",
            tool_results=tool_results,
            request_symbols=request_symbols,
        ),
        "risk_snapshot": _StaticTool(
            name="risk_feature",
            description="Read-only risk feature lookup for requested symbols. Input: symbol such as AAPL.",
            bundle_field="risk_snapshot",
            tool_results=tool_results,
            request_symbols=request_symbols,
        ),
        "portfolio_snapshot": _StaticTool(
            name="portfolio_snapshot",
            description="Read-only portfolio snapshot lookup. Input: optional symbol or empty string.",
            bundle_field="portfolio_snapshot",
            tool_results=tool_results,
            request_symbols=request_symbols,
        ),
    }


class StaticToolInput(BaseModel):
    symbol: str = Field(default="", description="Ticker symbol to retrieve, e.g. AAPL. Leave empty for portfolio-level tools.")


class _StaticTool(BaseTool):
    name: str
    description: str
    args_schema: type[BaseModel] = StaticToolInput
    bundle_field: str
    tool_results: ToolResultBundle
    request_symbols: list[str]

    def _run(self, symbol: str = "") -> str:
        result = getattr(self.tool_results, self.bundle_field)
        if result is None:
            payload = json.dumps(
                {
                    "tool": self.name,
                    "status": "UNAVAILABLE",
                    "symbol": symbol,
                    "error_message": f"{self.bundle_field} was not provided",
                }
            )
            _append_tool_trace(self.name, self.bundle_field, symbol, payload)
            return payload
        payload = _filtered_tool_payload(result, symbol or "")
        payload = _inject_trace_marker(payload, self.name)
        _append_tool_trace(self.name, self.bundle_field, symbol, payload)
        return payload


def _filtered_tool_payload(result: Any, symbol: str) -> str:
    payload = result.model_dump(mode="json")
    data = payload.get("data")
    if isinstance(data, dict):
        requested = symbol.strip().upper()
        if requested:
            payload["data"] = {requested: data[requested]} if requested in data else {}
        elif len(data) == 1:
            payload["data"] = data
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _inject_trace_marker(result_json: str, tool_name: str) -> str:
    marker = os.getenv("ORCA_CREWAI_TRACE_MARKER", "ORCA_TOOL_SEEN_MARKER")
    try:
        payload = json.loads(result_json)
    except json.JSONDecodeError:
        return result_json
    if isinstance(payload, dict):
        payload["trace_marker"] = f"{marker}:{tool_name}"
        refs = payload.get("source_refs")
        if isinstance(refs, list):
            payload["source_refs"] = [f"TRACE_MARKER:{marker}:{tool_name}", *refs]
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return result_json


def _append_tool_trace(tool_name: str, bundle_field: str, query: str, result_json: str) -> None:
    trace_path = Path("/tmp/orca_crewai_tool_calls.jsonl")
    try:
        parsed = json.loads(result_json)
    except json.JSONDecodeError:
        parsed = None
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "tool": tool_name,
        "bundle_field": bundle_field,
        "query": query,
        "result_len": len(result_json),
        "result_status": parsed.get("status") if isinstance(parsed, dict) else None,
        "source_refs": parsed.get("source_refs") if isinstance(parsed, dict) else None,
        "data_keys": sorted((parsed.get("data") or {}).keys()) if isinstance(parsed, dict) and isinstance(parsed.get("data"), dict) else [],
        "result_preview": result_json[:2000],
    }
    try:
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


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
