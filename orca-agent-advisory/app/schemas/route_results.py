from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ScreenCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    final_score: float | None = None
    predicted_direction: str | None = None
    as_of: str | None = None
    latest_price: float | None = None
    r1: float | None = None
    RVOL20: float | None = None
    RSI14: float | None = None
    risk_prob: float | None = None
    status: str | None = None
    warnings: list[str] = Field(default_factory=list)


class UniverseScreenResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[ScreenCandidate] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ComparisonRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    final_score: float | None = None
    predicted_direction: str | None = None
    rank: int | None = None
    latest_price: float | None = None
    r1: float | None = None
    RVOL20: float | None = None
    RSI14: float | None = None
    risk_prob: float | None = None
    as_of: str | None = None
    status: str | None = None
    warnings: list[str] = Field(default_factory=list)


class SymbolComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[ComparisonRow] = Field(default_factory=list)


class WatchlistItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    status: str = "reviewed"
    final_score: float | None = None
    predicted_direction: str | None = None
    latest_price: float | None = None
    r1: float | None = None
    RVOL20: float | None = None
    RSI14: float | None = None
    risk_prob: float | None = None
    as_of: str | None = None
    warnings: list[str] = Field(default_factory=list)


class WatchlistReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[WatchlistItem] = Field(default_factory=list)


class MarketBriefResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    leaders: list[ScreenCandidate] = Field(default_factory=list)


class DataDiagnosticsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnostics: dict[str, Any] = Field(default_factory=dict)


class StreamingPipelineStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    table: str
    status: str
    latest_timestamp: str | None = None
    row_count: int | None = None
    error: str | None = None


class StreamingPipelineHealthResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stages: list[StreamingPipelineStage] = Field(default_factory=list)


class StreamingFreshnessRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str | None = None
    table: str
    latest_timestamp: str | None = None
    lag_minutes: float | None = None
    status: str
    error: str | None = None


class StreamingFreshnessResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[StreamingFreshnessRow] = Field(default_factory=list)


class StreamingAlertRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str | None = None
    severity: str | None = None
    alert_type: str | None = None
    message: str | None = None
    timestamp: str | None = None


class StreamingAlertReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alerts: list[StreamingAlertRow] = Field(default_factory=list)


class StreamingSymbolMonitorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str | None = None
    freshness: list[StreamingFreshnessRow] = Field(default_factory=list)
    alerts: list[StreamingAlertRow] = Field(default_factory=list)


class StreamingFeatureDriftRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str | None = None
    feature: str
    streaming_value: float | None = None
    batch_value: float | None = None
    delta: float | None = None
    status: str
    error: str | None = None


class StreamingFeatureDriftResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[StreamingFeatureDriftRow] = Field(default_factory=list)


class StreamingIngestionLagRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str
    latest_timestamp: str | None = None
    lag_minutes: float | None = None
    status: str
    error: str | None = None


class StreamingIngestionLagResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[StreamingIngestionLagRow] = Field(default_factory=list)


class StreamingTopicSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str
    status: str
    partition_count: int | None = None
    latest_offsets: dict[str, int] = Field(default_factory=dict)
    consumer_lag: dict[str, int | None] | None = None
    sample: dict[str, Any] = Field(default_factory=dict)
    limitation: str | None = None
    error: str | None = None


class StreamingTopicInspectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: list[StreamingTopicSample] = Field(default_factory=list)


class StreamingQualityIncident(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str | None = None
    table: str
    incident_type: str
    message: str
    timestamp: str | None = None


class StreamingQualityIncidentsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incidents: list[StreamingQualityIncident] = Field(default_factory=list)


class PortfolioRebalanceChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    current_weight: float
    target_weight: float
    change: float


class PortfolioRebalanceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    changes: list[PortfolioRebalanceChange] = Field(default_factory=list)
    cash_target_weight: float = 0.0
    constraints: dict[str, Any] = Field(default_factory=dict)
    human_review_required: bool = True
    message: str


class BacktestAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backtest_spec: dict[str, Any] = Field(default_factory=dict)
    status: str
    limitation: str
    suggested_next_action: str
    metrics: dict[str, Any] | None = None
    trades_summary: dict[str, Any] | None = None
    equity_curve_sampled: list[dict[str, Any]] | None = None
    warnings: list[str] = Field(default_factory=list)
