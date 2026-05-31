from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.decision import DecisionRationale, PortfolioAllocation, PortfolioSummary
from app.schemas.enums import InvestmentHorizon, PortfolioAction, Recommendation


class ManagerSynthesisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    time_horizon: InvestmentHorizon
    proposed_recommendation: Recommendation | None = None
    proposed_portfolio_action: PortfolioAction | None = None
    decision_rationale: list[DecisionRationale] = Field(default_factory=list)
    supporting_signals: list[str] = Field(default_factory=list)
    conflicting_signals: list[str] = Field(default_factory=list)
    risk_warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    data_citations: list[str] = Field(default_factory=list)
    debate_applied: bool = False
    debate_summary: str | None = None
    bullish_critic_points: list[str] = Field(default_factory=list)
    bearish_critic_points: list[str] = Field(default_factory=list)
    portfolio_allocation: list[PortfolioAllocation] = Field(default_factory=list)
    portfolio_summary: PortfolioSummary | None = None

    @model_validator(mode="after")
    def require_actionable_draft(self) -> "ManagerSynthesisOutput":
        has_single_symbol_draft = self.proposed_recommendation is not None
        has_portfolio_draft = bool(self.portfolio_allocation) and self.portfolio_summary is not None
        if not has_single_symbol_draft and not has_portfolio_draft:
            raise ValueError(
                "manager synthesis must include either proposed_recommendation or portfolio draft"
            )
        return self
