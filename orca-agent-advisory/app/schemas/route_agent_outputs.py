from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RouteAgentResponseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    result_type: str = Field(min_length=1)
    result: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
