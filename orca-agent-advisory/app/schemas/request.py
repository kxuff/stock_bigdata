from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.enums import DecisionMode, InvestmentHorizon, RiskTolerance


class UserContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    investment_horizon: InvestmentHorizon = InvestmentHorizon.MEDIUM_TERM
    target_sectors: list[str] = Field(default_factory=list)
    excluded_symbols: list[str] = Field(default_factory=list)
    max_single_asset_weight: float = Field(default=40.0, ge=0.0, le=100.0)
    allow_cash_position: bool = True
    custom_constraints: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_sectors", "excluded_symbols")
    @classmethod
    def strip_blank_items(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("list values must be unique after trimming")
        return cleaned

    @field_validator("excluded_symbols")
    @classmethod
    def normalize_excluded_symbols(cls, values: list[str]) -> list[str]:
        return [value.upper() for value in values]


class AdvisoryDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    timestamp: datetime
    as_of_timestamp: datetime
    user_query: str = Field(min_length=1)
    decision_mode: DecisionMode
    symbols: list[str] = Field(min_length=1)
    user_context: UserContext = Field(default_factory=UserContext)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, values: list[str]) -> list[str]:
        normalized = [value.strip().upper() for value in values if value.strip()]
        if not normalized:
            raise ValueError("symbols must contain at least one non-empty symbol")
        if len(normalized) != len(set(normalized)):
            raise ValueError("symbols must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_request_shape(self) -> "AdvisoryDecisionRequest":
        if self.decision_mode == DecisionMode.SINGLE_SYMBOL_ADVISORY and len(self.symbols) != 1:
            raise ValueError("single_symbol_advisory requires exactly one symbol")

        excluded = set(self.user_context.excluded_symbols)
        requested = set(self.symbols)
        overlap = sorted(requested & excluded)
        if overlap:
            raise ValueError(f"symbols cannot include excluded symbols: {', '.join(overlap)}")

        if self.as_of_timestamp > self.timestamp:
            raise ValueError("as_of_timestamp cannot be later than timestamp")

        return self
