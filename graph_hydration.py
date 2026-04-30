from schema import ClaimGraph, ClaimNode, FactCard, make_id
from schema_draft import ClaimGraphDraft
from typing import Dict, List

def hydrate_claim_graph(draft: ClaimGraphDraft, facts: Dict[str, FactCard]) -> ClaimGraph:
    temp_to_claim = {}
    nodes = {}
    warnings = []
    fact_ids_set = set(facts.keys())

    for d in draft.drafts:
        claim_id = make_id()
        temp_to_claim[d.temp_id] = claim_id

        parent_claim_ids = []
        for p in d.parent_temp_ids:
            if p not in temp_to_claim:
                raise ValueError(f"Missing parent temp_id: {p}")
            parent_claim_ids.append(temp_to_claim[p])

        cleaned_fact_ids = []
        for fid in d.fact_ids:
            if fid in fact_ids_set:
                cleaned_fact_ids.append(fid)
            else:
                warnings.append(f"Invalid fact_id {fid} in claim {d.temp_id}")

        node = ClaimNode(
            claim_id=claim_id,
            content=d.content,
            claim_type=d.claim_type,
            parent_claim_ids=parent_claim_ids,
            fact_ids=cleaned_fact_ids,
            reasoning=d.reasoning,
            confidence=d.confidence
        )
        nodes[claim_id] = node

    return ClaimGraph(
        nodes=nodes,
        facts=facts,
        hydration_warnings=warnings
    )
