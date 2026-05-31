from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CriticOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stance: Literal["BULLISH", "BEARISH"]
    score: float = Field(ge=0.0, le=1.0)
    key_points: list[str] = Field(default_factory=list)
    recommendation_adjustment: Literal["KEEP", "DOWNGRADE", "UPGRADE"]
    confidence_adjustment: float = Field(ge=-0.3, le=0.3)
    citations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
