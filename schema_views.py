from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict
from schema import FactCard

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
            summary=fact.summary
        )
        for fact in facts.values()
    ]
