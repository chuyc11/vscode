"""Single Source of Truth (SSOT) for all data models.

Every node in the pipeline imports its data structures from this file.
All models use Pydantic v2 with extra="forbid" to reject unknown fields.
"""

from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from typing import List, Optional, Dict
from datetime import datetime, timezone
import uuid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FactCard — atomic unit of evidence
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """Named entity extracted by NLP."""
    model_config = ConfigDict(extra="forbid")
    text: str
    label: str  # PERSON, ORG, GPE, MONEY, DATE, etc.
    start: int
    end: int
    confidence: float = 1.0


class SourceMetadata(BaseModel):
    """知识溯源元数据 — 记录内参文档的出处信息。"""
    model_config = ConfigDict(extra="forbid")
    file_name: str
    file_path: str = ""
    page_number: Optional[int] = None
    chunk_index: int = 0


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
    entities: List[Entity] = Field(default_factory=list)
    source_metadata: Optional[SourceMetadata] = None


# ---------------------------------------------------------------------------
# ClaimNode / ClaimGraph — structured reasoning graph
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Attack / Assassination — adversarial review layer
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ClaimDraft / ClaimGraphDraft — D-agent draft output
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FactCardViewForD — serialised view of FactCard for D-agent LLM prompts
# ---------------------------------------------------------------------------

class FactCardViewForD(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact_id: str
    content: str
    source_tier: str
    evidence_type: str
    timestamp: str
    credibility_score: float
    relevance_score: float
    summary: str


# ---------------------------------------------------------------------------
# TickerMapping / QuantFactCard — C-node financial enrichment
# ---------------------------------------------------------------------------

class TickerMapping(BaseModel):
    """Entity → stock ticker mapping with context-aware validation."""
    model_config = ConfigDict(extra="forbid")
    entity_text: str
    ticker: str
    company_name: str = ""
    validation_confidence: float = 0.0  # 0-1, how confident the mapping is
    validation_reason: str = ""  # LLM reasoning for the mapping
    context_aligned: bool = False  # whether entity context matches ticker


class PriceData(BaseModel):
    """Point-in-time price snapshot from market data provider."""
    model_config = ConfigDict(extra="forbid")
    last_price: Optional[float] = None
    prev_close: Optional[float] = None
    open_price: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    currency: str = "USD"


class FundamentalData(BaseModel):
    """Key fundamental metrics from financial statements."""
    model_config = ConfigDict(extra="forbid")
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    revenue_ttm: Optional[float] = None
    net_income_ttm: Optional[float] = None
    eps_ttm: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    beta: Optional[float] = None


class QuantFactCard(BaseModel):
    """FactCard enriched with quantitative financial data from OpenBB."""
    model_config = ConfigDict(extra="forbid")
    quant_id: str = Field(default_factory=make_id)
    source_fact_id: str  # links back to original FactCard.fact_id
    ticker: str
    company_name: str = ""
    mapping_confidence: float = 0.0
    mapping_reason: str = ""
    price: PriceData = Field(default_factory=PriceData)
    fundamentals: FundamentalData = Field(default_factory=FundamentalData)
    data_source: str = "openbb"  # provenance
    fetch_timestamp: datetime = Field(default_factory=utc_now)
    fetch_success: bool = True
    fetch_errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# FactCardBatch — batch serialization wrapper for Redis storage
# ---------------------------------------------------------------------------

class FactCardBatch(BaseModel):
    """Batch wrapper for serializing a Dict[str, FactCard] to Redis."""
    model_config = ConfigDict(extra="forbid")
    facts: Dict[str, FactCard]


class QuantFactCardBatch(BaseModel):
    """Batch wrapper for serializing List[QuantFactCard] to Redis."""
    model_config = ConfigDict(extra="forbid")
    quant_facts: List["QuantFactCard"]


def build_fact_views_for_d(facts: Dict[str, FactCard]) -> List[FactCardViewForD]:
    return [
        FactCardViewForD(
            fact_id=fact.fact_id,
            content=fact.content,
            source_tier=fact.source_tier.value,
            evidence_type=fact.evidence_type.value,
            timestamp=fact.timestamp.isoformat(),
            credibility_score=fact.credibility_score,
            relevance_score=fact.relevance_score,
            summary=fact.summary,
        )
        for fact in facts.values()
    ]
