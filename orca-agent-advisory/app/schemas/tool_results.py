from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.enums import DecisionMode, RiskLabel, SentimentLabel, ToolStatus, ValuationLabel


class ToolResultValidationError(ValueError):
    """Raised when parsed tool results are insufficient for a request."""


class Freshness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_stale: bool
    last_updated_at: datetime
    max_age_seconds: int | None = Field(default=None, ge=0)


class BaseToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    status: ToolStatus
    request_id: str = Field(min_length=1)
    as_of_timestamp: datetime
    freshness: Freshness
    source_refs: list[str] = Field(default_factory=list)
    error_message: str | None = None

    @model_validator(mode="after")
    def require_error_message_for_failed_status(self) -> "BaseToolResult":
        if self.status in {ToolStatus.UNAVAILABLE, ToolStatus.ERROR} and not self.error_message:
            raise ValueError("error_message is required when tool status is unavailable or error")
        return self


class TechnicalIndicators(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rsi_14: float | None = Field(default=None, ge=0.0, le=100.0)
    macd_signal: str | None = None
    bollinger_position: str | None = None
    sma20_vs_price: str | None = None


class MarketFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latest_price: float = Field(gt=0.0)
    price_change_pct_1d: float
    volume_ratio_20d: float = Field(ge=0.0)
    trend_direction: str
    technical_indicators: TechnicalIndicators = Field(default_factory=TechnicalIndicators)


class MarketFeatureToolResult(BaseToolResult):
    data: dict[str, MarketFeature] = Field(default_factory=dict)


class MlPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_direction: str
    probability_up: float = Field(ge=0.0, le=1.0)
    probability_down: float = Field(ge=0.0, le=1.0)
    model_version: str
    feature_window: str | None = None


class MlPredictionToolResult(BaseToolResult):
    data: dict[str, MlPrediction] = Field(default_factory=dict)


class SentimentSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment_label: SentimentLabel
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    article_count: int = Field(ge=0)
    top_drivers: list[str] = Field(default_factory=list)
    latest_article_published_at: datetime | None = None
    oldest_article_published_at: datetime | None = None
    sentiment_scored_at: datetime | None = None
    stale_article_count: int | None = Field(default=None, ge=0)


class SentimentToolResult(BaseToolResult):
    data: dict[str, SentimentSnapshot] = Field(default_factory=dict)


class ValuationSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valuation_label: ValuationLabel
    pe_ratio: float | None = Field(default=None, gt=0.0)
    sector_pe_ratio: float | None = Field(default=None, gt=0.0)
    fair_value_estimate: float | None = Field(default=None, gt=0.0)
    upside_downside_pct: float | None = None
    valuation_method: str | None = None
    valuation_quality: str | None = None
    valuation_fetched_at: datetime | None = None
    fundamentals_as_of: datetime | None = None
    sector_sample_count: int | None = Field(default=None, ge=0)


class ValuationToolResult(BaseToolResult):
    data: dict[str, ValuationSnapshot] = Field(default_factory=dict)


class RiskSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_label: RiskLabel
    volatility_30d: float = Field(ge=0.0)
    max_drawdown_90d: float
    beta: float | None = None
    risk_factors: list[str] = Field(default_factory=list)
    confidence_cap: float = Field(ge=0.0, le=1.0)


class RiskToolResult(BaseToolResult):
    data: dict[str, RiskSnapshot] = Field(default_factory=dict)


class HoldingSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    weight_pct: float = Field(ge=0.0, le=100.0)
    market_value: float | None = Field(default=None, ge=0.0)


class PortfolioSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    holdings: list[HoldingSnapshot]
    cash_weight_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    constraints: dict[str, Any] = Field(default_factory=dict)


class PortfolioToolResult(BaseToolResult):
    data: PortfolioSnapshot | None = None


class ToolResultBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_features: MarketFeatureToolResult | None = None
    ml_predictions: MlPredictionToolResult | None = None
    sentiment_snapshot: SentimentToolResult | None = None
    valuation_snapshot: ValuationToolResult | None = None
    risk_snapshot: RiskToolResult | None = None
    portfolio_snapshot: PortfolioToolResult | None = None

    def validate_required_for(self, request: Any) -> None:
        self._require_tool(
            self.market_features,
            tool_name="market_features",
            required_path="MarketFeatureTool",
        )

        market_data = self.market_features.data if self.market_features else {}
        missing_market_symbols = [
            symbol for symbol in request.symbols if symbol not in market_data
        ]
        if missing_market_symbols:
            raise ToolResultValidationError(
                "market_features missing required symbols: " + ", ".join(missing_market_symbols)
            )

        if request.decision_mode == DecisionMode.PORTFOLIO_RECOMMENDATION:
            self._require_tool(
                self.risk_snapshot,
                tool_name="risk_snapshot",
                required_path="RiskFeatureTool",
            )
            self._require_tool(
                self.portfolio_snapshot,
                tool_name="portfolio_snapshot",
                required_path="PortfolioTool",
            )

    @staticmethod
    def _require_tool(result: BaseToolResult | None, *, tool_name: str, required_path: str) -> None:
        if result is None:
            raise ToolResultValidationError(f"{tool_name} is required: missing {required_path}")
        if result.status != ToolStatus.SUCCESS:
            raise ToolResultValidationError(
                f"{tool_name} is required but status is {result.status}: {result.error_message or required_path}"
            )
        if result.freshness.is_stale:
            updated_at = result.freshness.last_updated_at.isoformat()
            raise ToolResultValidationError(f"{tool_name} is stale as of {updated_at}")
