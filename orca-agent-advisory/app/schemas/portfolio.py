from pydantic import BaseModel, ConfigDict, Field


class PortfolioPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    quantity: float | None = None
    market_value: float | None = None
    weight: float | None = None
    currency: str | None = None


class PortfolioAccountSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_id: str
    tenant_id: str
    base_currency: str
    positions: list[PortfolioPosition] = Field(default_factory=list)
    cash: float
    as_of: str
    source: str
