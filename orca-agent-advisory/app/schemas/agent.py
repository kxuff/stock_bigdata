from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.enums import AgentRoute


class AgentContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str | None = None
    symbols: list[str] = Field(default_factory=list)
    watchlist: list[str] = Field(default_factory=list)
    universe: list[str] = Field(default_factory=list)
    risk_tolerance: str = "MODERATE"
    investment_horizon: str = "MEDIUM_TERM"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    context: AgentContext = Field(default_factory=AgentContext)


class SuggestedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    route: AgentRoute
    symbol: str | None = None
    symbols: list[str] = Field(default_factory=list)


class RoutedAgentQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: AgentRoute
    confidence: float = Field(ge=0.0, le=1.0)
    symbols: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    message: str = Field(min_length=1)
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)


class AgentQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: AgentRoute
    status: str
    message: str
    symbols: list[str] = Field(default_factory=list)
    result_type: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
    router_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
