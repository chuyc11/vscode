import pytest
import uuid
from unittest.mock import patch, MagicMock
from orchestrator import run_pipeline
from schema import FactCard, SourceTier, EvidenceType, FinalDecision, ClaimType
from schema import ClaimGraphDraft, ClaimDraft, FactCardBatch
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# In-memory mock store — replaces Redis + zstd for testing
# ---------------------------------------------------------------------------

class MockStateStore:
    """In-memory StateStore mock for testing (no Redis dependency)."""

    def __init__(self):
        self._store: dict[str, bytes] = {}

    def put(self, obj: BaseModel) -> str:
        key = f"pipeline:{uuid.uuid4().hex}"
        self._store[key] = obj.model_dump_json().encode("utf-8")
        return key

    def get(self, key: str, model_class: type) -> BaseModel:
        if key not in self._store:
            raise KeyError(f"Key missing: {key}")
        return model_class.model_validate_json(self._store[key])

    def cleanup(self, *keys: str) -> None:
        for k in keys:
            self._store.pop(k, None)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

fact1 = FactCard(fact_id="f1", content="Fact 1", source_tier=SourceTier.PRIMARY, evidence_type=EvidenceType.DOCUMENT)
facts = {"f1": fact1}

# Sample draft for PASS (has supporting fact, high confidence)
draft_pass = ClaimGraphDraft(drafts=[
    ClaimDraft(
        temp_id="c1", content="Claim 1", claim_type=ClaimType.FACT,
        fact_ids=["f1"], reasoning="Based on fact 1", confidence=0.9
    )
])

# Draft with no facts (triggers REJECT)
draft_reject = ClaimGraphDraft(drafts=[
    ClaimDraft(
        temp_id="c1", content="Unfounded claim", claim_type=ClaimType.FACT,
        fact_ids=[], reasoning="", confidence=0.3
    )
])


# ---------------------------------------------------------------------------
# Tests — mock StateStore + agent LLM calls
# ---------------------------------------------------------------------------

@patch('orchestrator._get_store', return_value=MockStateStore())
@patch('orchestrator.enrich_financial_data', return_value=[])
@patch('agent_d_strategist.call_llm_json')
@patch('agent_e_assassin.call_llm')
@patch('agent_g_briefing.call_llm')
def test_pass(mock_g_llm, mock_e_llm, mock_d_llm, mock_c, mock_store):
    mock_d_llm.return_value = draft_pass.model_dump_json()
    mock_e_llm.return_value = "[]"
    mock_g_llm.return_value = "Intelligence Briefing: All claims passed. Claim 1 is well supported."

    result = run_pipeline("topic", facts)
    assert result.status == "COMPLETED"
    assert result.final_decision == FinalDecision.PASS
    assert result.iteration_count >= 1
    assert "Claim 1" in result.briefing


@patch('orchestrator._get_store', return_value=MockStateStore())
@patch('orchestrator.enrich_financial_data', return_value=[])
@patch('agent_d_strategist.call_llm_json')
@patch('agent_e_assassin.call_llm')
@patch('agent_g_briefing.call_llm')
def test_reject_unfounded(mock_g_llm, mock_e_llm, mock_d_llm, mock_c, mock_store):
    mock_d_llm.return_value = draft_reject.model_dump_json()
    mock_e_llm.return_value = "[]"
    mock_g_llm.return_value = "Briefing: Rejected."

    result = run_pipeline("topic", facts)
    assert result.status == "COMPLETED"
    assert result.final_decision in (FinalDecision.REJECT, FinalDecision.PASSED_WITH_RISKS)
    assert result.briefing is not None


@patch('orchestrator._get_store', return_value=MockStateStore())
@patch('orchestrator.enrich_financial_data', return_value=[])
@patch('agent_d_strategist.call_llm_json')
@patch('agent_e_assassin.call_llm')
@patch('agent_g_briefing.call_llm')
def test_reinforcement_on_attack(mock_g_llm, mock_e_llm, mock_d_llm, mock_c, mock_store):
    """When E rejects, D should be called again to reinforce."""
    draft_json = draft_pass.model_dump_json()
    mock_d_llm.side_effect = [draft_json, draft_json]
    attack_finding = '[{"claim_id": "c1", "attack_type": "logical_leap", "severity": "high", "description": "Logic gap"}]'
    mock_e_llm.side_effect = [attack_finding, "[]"]
    mock_g_llm.return_value = "Briefing after reinforcement."

    result = run_pipeline("topic", facts)
    assert result.status in ("COMPLETED", "FAILED")
    assert result.iteration_count >= 1
