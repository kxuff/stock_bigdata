from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.enums import ToolStatus


class ToolCallAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    status: ToolStatus
    source_refs: list[str] = Field(default_factory=list)
    result_hash: str = Field(pattern=r"^sha256:.+")


class RetrievedToolAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_calls: list[ToolCallAudit] = Field(default_factory=list)
    tool_result_bundle_hash: str | None = Field(default=None, pattern=r"^sha256:.+")


class AuditMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    model_provider: str = "DeepSeek"
    model_name: str | None = None
    framework: str = "CrewAI"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    input_request_hash: str = Field(pattern=r"^sha256:.+")
    tool_result_bundle_hash: str = Field(pattern=r"^sha256:.+")
    validator_version: str = Field(min_length=1)
    created_at: datetime
