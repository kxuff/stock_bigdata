from typing import Iterable

from pydantic import ValidationError

from app.schemas.agent_outputs import AgentOutputBundle, BaseAgentOutput
from app.schemas.decision import PortfolioSummary
from app.schemas.enums import AgentStatus
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle
from app.validators.financial_validator import validate_financial_output


class DecisionValidationError(ValueError):
    """Raised when assembled advisory output fails deterministic validation."""


def validate_decision_output(decision: object, request: AdvisoryDecisionRequest) -> None:
    try:
        type(decision).model_validate(decision.model_dump(mode="json"))
    except ValidationError as exc:
        raise DecisionValidationError(str(exc)) from exc

    validation = validate_financial_output(decision.model_dump(mode="json"), request=request)
    if not validation.passed:
        raise DecisionValidationError("; ".join(validation.violations))


def agent_confidence(output: BaseAgentOutput | None) -> float | None:
    if output is None or output.status == AgentStatus.SKIPPED:
        return None
    return output.confidence


def ml_confidence(symbol: str, tool_results: ToolResultBundle) -> float | None:
    if tool_results.ml_predictions is None:
        return None
    prediction = tool_results.ml_predictions.data.get(symbol)
    if prediction is None:
        return None
    return max(prediction.probability_up, prediction.probability_down)


def collect_limitations(
    agent_outputs: AgentOutputBundle,
    synthesis,
    quality_warnings: Iterable[str],
) -> list[str]:
    return unique([*synthesis.limitations, *agent_limitations(agent_outputs), *quality_warnings])


def agent_limitations(agent_outputs: AgentOutputBundle) -> list[str]:
    limitations: list[str] = []
    for output in [
        agent_outputs.market_data_agent,
        agent_outputs.sentiment_agent,
        agent_outputs.valuation_agent,
        agent_outputs.risk_agent,
    ]:
        if output is None:
            continue
        limitations.extend(output.limitations)
        limitations.extend(f"missing:{field}" for field in output.missing_fields)
    return unique(limitations)


def tool_citations(tool_results: ToolResultBundle) -> list[str]:
    citations: list[str] = []
    for result in [
        tool_results.market_features,
        tool_results.ml_predictions,
        tool_results.sentiment_snapshot,
        tool_results.valuation_snapshot,
        tool_results.risk_snapshot,
        tool_results.portfolio_snapshot,
    ]:
        if result is not None:
            citations.extend(result.source_refs)
    return unique(citations)


def portfolio_summary(agent_outputs: AgentOutputBundle) -> PortfolioSummary:
    return PortfolioSummary(
        expected_risk_label=agent_outputs.risk_agent.risk_label,
        concentration_risk=agent_outputs.risk_agent.risk_label,
        dominant_themes=unique(agent_outputs.risk_agent.risk_factors[:3]),
    )


def unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
