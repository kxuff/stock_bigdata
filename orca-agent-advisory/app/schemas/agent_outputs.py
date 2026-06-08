from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.enums import AgentStatus, RiskLabel, SentimentLabel, SignalStance, ValuationLabel


class BaseAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AgentStatus
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    missing_fields: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_limitation_when_skipped(self) -> "BaseAgentOutput":
        if self.status == AgentStatus.SKIPPED and not self.limitations:
            raise ValueError("skipped agent outputs must include at least one limitation")
        return self


class MarketSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    stance: SignalStance
    confidence: float = Field(ge=0.0, le=1.0)
    drivers: list[str] = Field(default_factory=list)


class MarketDataAgentOutput(BaseAgentOutput):
    market_signals: list[MarketSignal] = Field(default_factory=list)
    ml_signal_available: bool = True


class SentimentAgentOutput(BaseAgentOutput):
    sentiment_label: SentimentLabel
    sentiment_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    top_drivers: list[str] = Field(default_factory=list)


class ValuationAgentOutput(BaseAgentOutput):
    valuation_label: ValuationLabel
    valuation_drivers: list[str] = Field(default_factory=list)


class RiskAgentOutput(BaseAgentOutput):
    risk_label: RiskLabel
    risk_factors: list[str] = Field(default_factory=list)
    confidence_cap: float = Field(ge=0.0, le=1.0)


class AgentOutputBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_data_agent: MarketDataAgentOutput
    sentiment_agent: SentimentAgentOutput | None = None
    valuation_agent: ValuationAgentOutput | None = None
    risk_agent: RiskAgentOutput
