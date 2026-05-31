import os

from app.application.decision.decision_helpers import agent_limitations, portfolio_summary, tool_citations
from app.application.ports.crew_orchestrator import CrewOrchestratedOutputs
from app.application.specialists import analyze_market_data, analyze_risk, analyze_sentiment, analyze_valuation
from app.schemas.agent_outputs import AgentOutputBundle
from app.schemas.decision import DecisionRationale, PortfolioAllocation
from app.schemas.enums import DecisionMode, FactorWeight, PortfolioAction, Recommendation, RiskLabel, SignalStance
from app.schemas.manager_outputs import ManagerSynthesisOutput
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle


def pytest_configure() -> None:
    os.environ.setdefault("OPENAI_API_KEY", "test-openai-api-key")


def fixture_agent_outputs(
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
) -> AgentOutputBundle:
    return AgentOutputBundle(
        market_data_agent=analyze_market_data(request, tool_results),
        sentiment_agent=analyze_sentiment(request, tool_results),
        valuation_agent=analyze_valuation(request, tool_results),
        risk_agent=analyze_risk(request, tool_results),
    )


def fixture_manager_synthesis(
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
    agent_outputs: AgentOutputBundle,
) -> ManagerSynthesisOutput:
    if request.decision_mode == DecisionMode.PORTFOLIO_RECOMMENDATION:
        cash_weight = float(request.user_context.custom_constraints.get("min_cash_weight", 0.0))
        cash_weight = cash_weight if request.user_context.allow_cash_position else 0.0
        symbol_weight = round((100.0 - cash_weight) / len(request.symbols), 2)
        allocations = [
            PortfolioAllocation(
                symbol=symbol,
                weight_pct=symbol_weight,
                portfolio_action=PortfolioAction.MAINTAIN_WEIGHT,
                rationale="Maintain exposure for fixture.",
            )
            for symbol in request.symbols
        ]
        if request.user_context.allow_cash_position:
            allocations.append(
                PortfolioAllocation(
                    symbol="CASH",
                    weight_pct=round(100.0 - sum(item.weight_pct for item in allocations), 2),
                    portfolio_action=PortfolioAction.CASH_BUFFER,
                    rationale="Cash buffer satisfies fixture constraints.",
                )
            )
        else:
            allocations[-1].weight_pct += round(100.0 - sum(item.weight_pct for item in allocations), 2)
        return ManagerSynthesisOutput(
            summary="Fixture portfolio draft.",
            time_horizon=request.user_context.investment_horizon,
            proposed_portfolio_action=PortfolioAction.MAINTAIN_WEIGHT,
            portfolio_allocation=allocations,
            portfolio_summary=portfolio_summary(agent_outputs),
            risk_warnings=agent_outputs.risk_agent.risk_factors,
            limitations=agent_limitations(agent_outputs),
            data_citations=tool_citations(tool_results),
        )

    symbol = request.symbols[0]
    market_signal = agent_outputs.market_data_agent.market_signals[0]
    risk_label = agent_outputs.risk_agent.risk_label
    recommendation = Recommendation.HOLD if risk_label in {RiskLabel.HIGH, RiskLabel.CRITICAL} else Recommendation.BUY
    return ManagerSynthesisOutput(
        summary=f"{symbol} fixture draft.",
        time_horizon=request.user_context.investment_horizon,
        proposed_recommendation=recommendation,
        decision_rationale=[
            DecisionRationale(
                factor="market_signal",
                stance=market_signal.stance,
                weight=FactorWeight.HIGH,
                explanation=agent_outputs.market_data_agent.summary,
            ),
            DecisionRationale(
                factor="risk",
                stance=SignalStance.BEARISH if risk_label in {RiskLabel.HIGH, RiskLabel.CRITICAL} else SignalStance.NEUTRAL,
                weight=FactorWeight.HIGH,
                explanation=agent_outputs.risk_agent.summary,
            ),
        ],
        supporting_signals=market_signal.drivers,
        conflicting_signals=["Bullish market signal conflicts with high risk."] if risk_label in {RiskLabel.HIGH, RiskLabel.CRITICAL} else [],
        risk_warnings=agent_outputs.risk_agent.risk_factors,
        limitations=agent_limitations(agent_outputs),
        data_citations=tool_citations(tool_results),
    )


class FixtureCrewRunner:
    def run_orchestrated(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
    ) -> CrewOrchestratedOutputs:
        agent_outputs = fixture_agent_outputs(request, tool_results)
        return CrewOrchestratedOutputs(
            agent_outputs=agent_outputs,
            manager_payload=fixture_manager_synthesis(request, tool_results, agent_outputs),
        )
