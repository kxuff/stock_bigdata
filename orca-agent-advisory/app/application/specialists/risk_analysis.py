from app.schemas.agent_outputs import RiskAgentOutput
from app.schemas.enums import AgentStatus, DecisionMode, RiskLabel, ToolStatus
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle


def analyze_risk(
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
) -> RiskAgentOutput:
    risk_result = tool_results.risk_snapshot
    if risk_result is None or risk_result.status != ToolStatus.SUCCESS:
        return RiskAgentOutput(
            status=AgentStatus.ERROR,
            summary="Required risk context is unavailable.",
            confidence=0.0,
            missing_fields=["risk_snapshot"],
            limitations=["RISK_CONTEXT_UNAVAILABLE"],
            source_refs=[],
            risk_label=RiskLabel.HIGH,
            risk_factors=["missing required risk context"],
            confidence_cap=0.45,
        )

    source_refs = list(risk_result.source_refs)
    missing_fields: list[str] = []
    risk_labels: list[RiskLabel] = []
    risk_factors: list[str] = []
    confidence_caps: list[float] = []
    for symbol in request.symbols:
        snapshot = risk_result.data.get(symbol)
        if snapshot is None:
            missing_fields.append(f"risk_snapshot.{symbol}")
            continue
        risk_labels.append(snapshot.risk_label)
        risk_factors.extend(snapshot.risk_factors)
        confidence_caps.append(snapshot.confidence_cap)
        risk_factors.append(f"{symbol} volatility_30d={snapshot.volatility_30d}")
        risk_factors.append(f"{symbol} max_drawdown_90d={snapshot.max_drawdown_90d}")

    if request.decision_mode == DecisionMode.PORTFOLIO_RECOMMENDATION:
        portfolio_result = tool_results.portfolio_snapshot
        if portfolio_result is None or portfolio_result.status != ToolStatus.SUCCESS:
            missing_fields.append("portfolio_snapshot")
            risk_factors.append("portfolio snapshot unavailable for portfolio mode")
            if portfolio_result is not None:
                source_refs.extend(portfolio_result.source_refs)
        else:
            source_refs.extend(portfolio_result.source_refs)
            portfolio = portfolio_result.data
            if portfolio is not None:
                overweight = [
                    holding.symbol
                    for holding in portfolio.holdings
                    if holding.weight_pct > request.user_context.max_single_asset_weight
                ]
                if overweight:
                    risk_factors.append(
                        "single asset weight exceeds user constraint: " + ", ".join(overweight)
                    )

    if not risk_labels:
        return RiskAgentOutput(
            status=AgentStatus.ERROR,
            summary="Risk snapshots did not include requested symbols.",
            confidence=0.0,
            missing_fields=missing_fields,
            limitations=["RISK_SYMBOL_CONTEXT_UNAVAILABLE"],
            source_refs=source_refs,
            risk_label=RiskLabel.HIGH,
            risk_factors=risk_factors,
            confidence_cap=0.45,
        )

    risk_label = _max_risk_label(risk_labels)
    confidence_cap = min(confidence_caps) if confidence_caps else _default_cap(risk_label)
    return RiskAgentOutput(
        status=AgentStatus.DEGRADED if missing_fields else AgentStatus.SUCCESS,
        summary="Risk was summarized from RiskFeatureTool and portfolio constraints.",
        confidence=round(min(0.8, confidence_cap), 2),
        missing_fields=missing_fields,
        limitations=["PORTFOLIO_CONTEXT_UNAVAILABLE"] if "portfolio_snapshot" in missing_fields else [],
        source_refs=source_refs,
        risk_label=risk_label,
        risk_factors=list(dict.fromkeys(risk_factors)),
        confidence_cap=confidence_cap,
    )


def _max_risk_label(labels: list[RiskLabel]) -> RiskLabel:
    severity = {
        RiskLabel.LOW: 0,
        RiskLabel.MEDIUM: 1,
        RiskLabel.HIGH: 2,
        RiskLabel.CRITICAL: 3,
    }
    return max(labels, key=lambda label: severity[label])


def _default_cap(risk_label: RiskLabel) -> float:
    return {
        RiskLabel.LOW: 0.95,
        RiskLabel.MEDIUM: 0.85,
        RiskLabel.HIGH: 0.65,
        RiskLabel.CRITICAL: 0.45,
    }[risk_label]
