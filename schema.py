from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from typing import List, Optional, Dict
from datetime import datetime, timezone
import uuid

def make_id() -> str:
    return str(uuid.uuid4())

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class SourceTier(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"

class EvidenceType(Enum):
    DOCUMENT = "document"
    WITNESS = "witness"
    DIGITAL = "digital"
    PHYSICAL = "physical"

class ClaimType(Enum):
    FACT = "fact"
    OPINION = "opinion"
    PREDICTION = "prediction"

class AttackType(Enum):
    CONTRADICTION = "contradiction"
    INCONSISTENCY = "inconsistency"
    FABRICATION = "fabrication"
    MISATTRIBUTION = "misattribution"
    LOGICAL_LEAP = "logical_leap"
    SELECTION_BIAS = "selection_bias"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class AttackStatus(Enum):
    UNATTACKED = "unattacked"
    ATTACKED = "attacked"
    DEFENDED = "defended"

class FinalDecision(Enum):
    PASS = "pass"
    PASSED_WITH_RISKS = "passed_with_risks"
    FAILED = "failed"
    REJECT = "reject"

class FactCard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact_id: str = Field(default_factory=make_id)
    content: str
    source_tier: SourceTier
    evidence_type: EvidenceType
    timestamp: datetime = Field(default_factory=utc_now)
    credibility_score: float = 0.7
    relevance_score: float = 0.5
    summary: str = ""

class ClaimNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str = Field(default_factory=make_id)
    content: str
    claim_type: ClaimType
    parent_claim_ids: List[str] = Field(default_factory=list)
    fact_ids: List[str] = Field(default_factory=list)
    attack_status: AttackStatus = AttackStatus.UNATTACKED
    risk_notes: List[str] = Field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.7

class ClaimGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")
    graph_id: str = Field(default_factory=make_id)
    nodes: Dict[str, ClaimNode]
    facts: Dict[str, FactCard]
    hydration_warnings: List[str] = Field(default_factory=list)

class AttackFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    attack_type: AttackType
    severity: Severity
    description: str
    evidence_quote: Optional[str] = None

class AssassinationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    report_id: str = Field(default_factory=make_id)
    findings: List[AttackFinding]
    final_decision: FinalDecision
    summary: str
