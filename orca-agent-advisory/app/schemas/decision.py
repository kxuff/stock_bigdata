from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.agent_outputs import AgentOutputBundle
from app.schemas.audit import AuditMetadata, RetrievedToolAudit
from app.schemas.enums import (
    ConflictLevel,
    DecisionMode,
    FactorWeight,
    InvestmentHorizon,
    PortfolioAction,
    Recommendation,
    ReviewReason,
    RiskLabel,
    RiskTolerance,
    SignalStance,
)


class ConfidenceBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_confidence: float = Field(ge=0.0, le=1.0)
    risk_adjusted_confidence: float = Field(ge=0.0, le=1.0)
    risk_cap: float = Field(ge=0.0, le=1.0)
    source_quality_cap: float = Field(ge=0.0, le=1.0)
    market_confidence: float = Field(ge=0.0, le=1.0)
    ml_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sentiment_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    valuation_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_adjustment: float = Field(ge=-1.0, le=1.0)
    source_quality_adjustment: float = Field(ge=-1.0, le=1.0)


class DecisionRationale(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor: str
    stance: SignalStance
    weight: FactorWeight
    explanation: str


class SourceQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_quality_score: float = Field(ge=0.0, le=1.0)
    freshness_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    violations: list[str] = Field(default_factory=list)


class PortfolioAllocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    weight_pct: float = Field(ge=0.0, le=100.0)
    portfolio_action: PortfolioAction
    rationale: str


class PortfolioSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_risk_label: RiskLabel
    concentration_risk: RiskLabel
    dominant_themes: list[str] = Field(default_factory=list)


class FinalDecisionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    run_id: str
    decision_mode: DecisionMode
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_breakdown: ConfidenceBreakdown
    requires_human_review: bool
    review_reasons: list[ReviewReason] = Field(default_factory=list)
    audit: AuditMetadata
    retrieved_tool_audit: RetrievedToolAudit
    data_citations: list[str] = Field(default_factory=list)
    not_financial_advice: Literal[True]

    @model_validator(mode="after")
    def align_audit_identifiers(self) -> "FinalDecisionBase":
        if self.audit.request_id != self.request_id:
            raise ValueError("audit.request_id must match request_id")
        if self.audit.run_id != self.run_id:
            raise ValueError("audit.run_id must match run_id")
        return self


class SingleSymbolDecision(FinalDecisionBase):
    decision_mode: Literal[DecisionMode.SINGLE_SYMBOL_ADVISORY]
    symbol: str
    recommendation: Recommendation
    time_horizon: InvestmentHorizon
    summary: str
    agent_outputs: AgentOutputBundle
    decision_rationale: list[DecisionRationale] = Field(default_factory=list)
    supporting_signals: list[str] = Field(default_factory=list)
    conflicting_signals: list[str] = Field(default_factory=list)
    conflict_level: ConflictLevel = ConflictLevel.NONE
    risk_warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    source_quality: SourceQuality


class PortfolioDecision(FinalDecisionBase):
    decision_mode: Literal[DecisionMode.PORTFOLIO_RECOMMENDATION]
    risk_profile: RiskTolerance
    portfolio_allocation: list[PortfolioAllocation] = Field(min_length=1)
    portfolio_summary: PortfolioSummary
    reasoning_trace: list[str] = Field(default_factory=list)
    validation_result: ValidationResult

    @model_validator(mode="after")
    def validate_allocation_total(self) -> "PortfolioDecision":
        total_weight = sum(allocation.weight_pct for allocation in self.portfolio_allocation)
        if abs(total_weight - 100.0) > 0.01:
            raise ValueError(f"portfolio_allocation weight_pct must total 100, got {total_weight}")
        return self


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: Literal["ERROR"]
    error_code: str
    message: str
    recoverable: bool
    missing_tool_results: list[str] = Field(default_factory=list)
