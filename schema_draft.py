from pydantic import BaseModel, ConfigDict, Field
from typing import List
from schema import ClaimType

class ClaimDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temp_id: str
    content: str
    claim_type: ClaimType
    parent_temp_ids: List[str] = Field(default_factory=list)
    fact_ids: List[str] = Field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.7

class ClaimGraphDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drafts: List[ClaimDraft]
