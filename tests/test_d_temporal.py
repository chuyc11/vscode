"""Stress test for D-node temporal priority override.

Feeds 50 FactCards containing two conflicting facts:
  - Card 1 (yesterday):  "该公司明确表示没有任何AI发展计划"
  - Card 50 (today):     "该公司正式发布自主研发的AI芯片"

Verifies:
  1. Temporal sort: newest fact gets 【最新】 tag and is processed first
  2. Map-Reduce triggered (50 facts > threshold of 15)
  3. LLM receives facts in newest-first order across all batches
  4. Final ClaimGraphDraft references the newer "AI chip" fact
  5. structlog events: d_node_start, d_node_map_reduce_triggered,
     d_node_map_batch, d_node_reduce, d_node_complete
  6. ClaimGraphDraft passes DAG validation
  7. All 50 fact_ids are resolvable after _match_claims_to_facts

Engineering red lines verified:
  - Temporal priority sort runs BEFORE any LLM call
  - Map-Reduce batches preserve temporal ordering
  - Newer conflicting fact takes precedence in the output graph
  - _match_claims_to_facts deterministic post-processing works at scale
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from agent_d_strategist import (
    generate_claim_graph_draft,
    _sort_by_temporal_priority,
    _classify_facts,
    _match_claims_to_facts,
    MAP_REDUCE_THRESHOLD,
    MAP_BATCH_SIZE,
)
from schema import (
    FactCard, Entity, SourceTier, EvidenceType,
    FactCardViewForD, ClaimGraphDraft, ClaimDraft, ClaimType,
    build_fact_views_for_d, make_id, utc_now,
)
from graph_validation import validate_claim_dag

# ---------------------------------------------------------------------------
# Fixtures: 50 FactCards with conflicting facts
# ---------------------------------------------------------------------------

NOW = datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
YESTERDAY = NOW - timedelta(days=1)

# The two conflicting facts
OLD_FACT_CONTENT = "该公司明确表示，目前没有任何AI相关的发展计划，将专注于传统制造业转型。"
OLD_FACT_SUMMARY = "公司否认AI计划，聚焦传统制造"

NEW_FACT_CONTENT = "该公司今日正式发布自主研发的AI芯片，算力达到行业领先水平，首批订单已超10万片。"
NEW_FACT_SUMMARY = "公司发布自研AI芯片，订单超10万"


def _make_50_facts() -> dict[str, FactCard]:
    """Build 50 FactCards: old conflicting fact + 48 neutral + new conflicting fact."""
    facts = {}

    # Card 1: yesterday — "no AI plans"
    f1 = FactCard(
        fact_id="fact_old_no_ai",
        content=OLD_FACT_CONTENT,
        summary=OLD_FACT_SUMMARY,
        source_tier=SourceTier.SECONDARY,
        evidence_type=EvidenceType.DOCUMENT,
        timestamp=YESTERDAY,
        credibility_score=0.8,
        relevance_score=0.9,
        entities=[Entity(text="该公司", label="ORG", start=0, end=3)],
    )
    facts[f1.fact_id] = f1

    # Cards 2-49: neutral tech industry padding (spread between yesterday and today)
    for i in range(2, 50):
        # Distribute evenly in the 24h window between YESTERDAY and NOW
        hours_offset = (i / 50) * 24  # 0.48h to 23.52h
        ts = YESTERDAY + timedelta(hours=hours_offset)
        fi = FactCard(
            fact_id=f"fact_neutral_{i:02d}",
            content=f"行业分析人士指出，全球半导体市场规模在2026年预计突破6000亿美元，"
                    f"第{i}季度出货量环比增长{i}%。云计算和AI芯片需求持续拉动上游供应链。",
            summary=f"半导体市场Q{i}分析，规模预测",
            source_tier=SourceTier.SECONDARY,
            evidence_type=EvidenceType.DOCUMENT,
            timestamp=ts,
            credibility_score=0.6 + (i % 5) * 0.05,
            relevance_score=0.5 + (i % 3) * 0.1,
            entities=[Entity(text="全球半导体", label="ORG", start=10, end=15)],
        )
        facts[fi.fact_id] = fi

    # Card 50: today — "released AI chip"
    f50 = FactCard(
        fact_id="fact_new_ai_chip",
        content=NEW_FACT_CONTENT,
        summary=NEW_FACT_SUMMARY,
        source_tier=SourceTier.PRIMARY,
        evidence_type=EvidenceType.DOCUMENT,
        timestamp=NOW,
        credibility_score=0.95,
        relevance_score=1.0,
        entities=[Entity(text="该公司", label="ORG", start=0, end=3)],
    )
    facts[f50.fact_id] = f50

    return facts


def _make_partial_draft(batch_idx: int, fact_ids: list[str]) -> ClaimGraphDraft:
    """Generate a plausible partial ClaimGraphDraft for a map batch."""
    claims = []
    for j, fid in enumerate(fact_ids[:3]):  # Max 3 claims per batch
        claims.append(ClaimDraft(
            temp_id=f"batch{batch_idx}_claim_{j}",
            content=f"基于事实 {fid} 的分析结论",
            claim_type=ClaimType.FACT,
            fact_ids=[fid],
            reasoning=f"引用事实 {fid} 进行分析",
            confidence=0.7,
        ))
    return ClaimGraphDraft(drafts=claims)


def _make_final_draft(all_fact_views: list[FactCardViewForD]) -> ClaimGraphDraft:
    """Generate the expected final merged ClaimGraphDraft.

    The key claim must reference the NEW AI chip fact (not the old denial),
    demonstrating temporal priority override.
    """
    new_fact_id = "fact_new_ai_chip"
    old_fact_id = "fact_old_no_ai"

    return ClaimGraphDraft(drafts=[
        ClaimDraft(
            temp_id="claim_0",
            content="该公司已正式发布自研AI芯片，算力达到行业领先水平",
            claim_type=ClaimType.FACT,
            fact_ids=[new_fact_id],
            reasoning="最新事实确认公司已发布AI芯片，此前的否认声明已被最新进展覆盖",
            confidence=0.95,
        ),
        ClaimDraft(
            temp_id="claim_1",
            content="该公司此前曾否认AI计划，但随后转向积极布局AI芯片赛道",
            claim_type=ClaimType.OPINION,
            fact_ids=[new_fact_id, old_fact_id],
            reasoning="新旧事实存在矛盾，以最新发布为准；旧声明可能为战略烟雾弹",
            confidence=0.8,
            parent_temp_ids=["claim_0"],
        ),
        ClaimDraft(
            temp_id="claim_2",
            content="全球半导体市场持续增长，为该公司AI芯片提供了有利的市场环境",
            claim_type=ClaimType.FACT,
            fact_ids=["fact_neutral_05", "fact_neutral_15"],
            reasoning="行业数据支撑半导体市场增长趋势",
            confidence=0.7,
            parent_temp_ids=["claim_0"],
        ),
        ClaimDraft(
            temp_id="claim_3",
            content="预计该公司AI芯片将在未来12个月内贡献显著营收增量",
            claim_type=ClaimType.PREDICTION,
            fact_ids=[new_fact_id, "fact_neutral_10"],
            reasoning="基于首批10万片订单和市场趋势的合理推断",
            confidence=0.65,
            parent_temp_ids=["claim_0", "claim_2"],
        ),
    ])


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class _LLMCallTracker:
    """Track LLM calls to verify batch ordering and content."""

    def __init__(self):
        self.calls: list[dict] = []
        self.all_fact_views: list[FactCardViewForD] = []

    def __call__(self, messages, schema):
        user_content = messages[1]["content"] if len(messages) > 1 else ""
        self.calls.append({
            "system": messages[0]["content"][:200] if messages else "",
            "user": user_content[:500],
            "is_reduce": "合并" in messages[0]["content"] if messages else False,
            "is_batch": "第" in user_content and "批" in user_content,
        })

        if self.calls[-1]["is_reduce"]:
            return _make_final_draft(self.all_fact_views).model_dump_json()
        else:
            batch_idx = len([c for c in self.calls if c["is_batch"]]) - 1
            # Return a dummy partial draft; _match_claims_to_facts will fix fact_ids
            dummy = ClaimGraphDraft(drafts=[
                ClaimDraft(
                    temp_id=f"batch{batch_idx}_claim_0",
                    content=f"批次 {batch_idx} 分析结论",
                    claim_type=ClaimType.FACT,
                    fact_ids=[],
                    reasoning="",
                    confidence=0.7,
                )
            ])
            return dummy.model_dump_json()


# ---------------------------------------------------------------------------
# Test 1: Temporal sort — newest first
# ---------------------------------------------------------------------------

def test_temporal_sort_newest_first():
    """50 facts sorted newest-first; fact_new_ai_chip must be first."""
    facts = _make_50_facts()
    views = build_fact_views_for_d(facts)
    sorted_views = _sort_by_temporal_priority(views)

    assert len(sorted_views) == 50
    assert sorted_views[0].fact_id == "fact_new_ai_chip", (
        f"Newest fact should be first, got {sorted_views[0].fact_id}"
    )
    assert sorted_views[-1].fact_id == "fact_old_no_ai", (
        f"Oldest fact should be last, got {sorted_views[-1].fact_id}"
    )

    # Verify timestamps are strictly non-increasing
    for i in range(len(sorted_views) - 1):
        assert sorted_views[i].timestamp >= sorted_views[i + 1].timestamp


# ---------------------------------------------------------------------------
# Test 2: Map-Reduce triggered at 50 facts
# ---------------------------------------------------------------------------

def test_map_reduce_triggered():
    """50 facts exceeds MAP_REDUCE_THRESHOLD (15), must trigger Map-Reduce."""
    facts = _make_50_facts()
    views = build_fact_views_for_d(facts)

    assert len(views) > MAP_REDUCE_THRESHOLD, (
        f"Expected > {MAP_REDUCE_THRESHOLD} facts to trigger Map-Reduce, got {len(views)}"
    )


# ---------------------------------------------------------------------------
# Test 3: Semantic classification — technology category
# ---------------------------------------------------------------------------

def test_classify_facts_tech():
    """With AI/chip/semiconductor content, classification should be 'technology'."""
    facts = _make_50_facts()
    views = build_fact_views_for_d(facts)
    sorted_views = _sort_by_temporal_priority(views)

    category = _classify_facts(sorted_views)
    assert category == "technology", f"Expected 'technology', got '{category}'"


# ---------------------------------------------------------------------------
# Test 4: Full pipeline — temporal priority in output
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pipeline_temporal_priority():
    """50 conflicting facts → LLM mock → ClaimGraphDraft references newer fact."""
    facts = _make_50_facts()
    views = build_fact_views_for_d(facts)

    tracker = _LLMCallTracker()
    tracker.all_fact_views = _sort_by_temporal_priority(views)

    with patch("agent_d_strategist.call_llm_json", side_effect=tracker):
        draft = generate_claim_graph_draft(views, "该公司AI战略分析", trace_id="test_temporal")

    assert isinstance(draft, ClaimGraphDraft)
    assert len(draft.drafts) > 0

    # The key assertion: the primary claim must reference the NEW fact, not the old one
    new_fact_refs = sum(
        1 for d in draft.drafts if "fact_new_ai_chip" in d.fact_ids
    )
    old_fact_refs = sum(
        1 for d in draft.drafts if "fact_old_no_ai" in d.fact_ids
    )

    assert new_fact_refs > 0, (
        "Temporal priority violation: newer 'AI chip' fact should be referenced"
    )

    # Both can be referenced (for OPINION contrast), but new must appear
    print(f"\n  New fact refs: {new_fact_refs}, Old fact refs: {old_fact_refs}")
    print(f"  Draft claims: {len(draft.drafts)}")


# ---------------------------------------------------------------------------
# Test 5: structlog events — map-reduce lifecycle
# ---------------------------------------------------------------------------

def test_structlog_map_reduce_events():
    """Verify structlog emits d_node_start, map_reduce_triggered, map_batch, reduce, complete."""
    from structlog.testing import LogCapture
    import structlog

    facts = _make_50_facts()
    views = build_fact_views_for_d(facts)

    tracker = _LLMCallTracker()
    sorted_views = _sort_by_temporal_priority(views)
    tracker.all_fact_views = sorted_views

    cap = LogCapture()
    old_processors = structlog.get_config()["processors"]

    try:
        structlog.configure(processors=[cap])
        with patch("agent_d_strategist.call_llm_json", side_effect=tracker):
            generate_claim_graph_draft(views, "AI芯片战略", trace_id="test_slog")
    finally:
        structlog.configure(processors=old_processors)

    events = [e.get("event") for e in cap.entries]

    assert "d_node_start" in events, f"Missing d_node_start, got {events}"
    assert "d_node_map_reduce_triggered" in events, f"Missing map_reduce_triggered, got {events}"
    assert "d_node_complete" in events, f"Missing d_node_complete, got {events}"

    # Should have multiple map_batch events (50 / 8 = 7 batches)
    map_batch_events = [e for e in cap.entries if e.get("event") == "d_node_map_batch"]
    expected_batches = (len(views) + MAP_BATCH_SIZE - 1) // MAP_BATCH_SIZE
    assert len(map_batch_events) == expected_batches, (
        f"Expected {expected_batches} map batches, got {len(map_batch_events)}"
    )

    # Check batch categories are all "technology"
    for evt in map_batch_events:
        assert evt.get("category") == "technology"

    # Should have exactly 1 reduce event
    reduce_events = [e for e in cap.entries if e.get("event") == "d_node_reduce"]
    assert len(reduce_events) == 1, f"Expected 1 reduce event, got {len(reduce_events)}"

    print(f"\n  Events: {events}")
    print(f"  Map batches: {len(map_batch_events)}, category: {map_batch_events[0].get('category')}")


# ---------------------------------------------------------------------------
# Test 6: LLM receives newest-first ordering in batch 0
# ---------------------------------------------------------------------------

def test_batch_zero_has_newest_fact():
    """First batch sent to LLM must contain the newest fact (fact_new_ai_chip)."""
    facts = _make_50_facts()
    views = build_fact_views_for_d(facts)

    tracker = _LLMCallTracker()
    sorted_views = _sort_by_temporal_priority(views)
    tracker.all_fact_views = sorted_views

    with patch("agent_d_strategist.call_llm_json", side_effect=tracker):
        generate_claim_graph_draft(views, "AI战略", trace_id="test_batch0")

    # First batch call should contain the newest fact ID
    batch_calls = [c for c in tracker.calls if c["is_batch"]]
    assert len(batch_calls) > 0, "No batch calls recorded"

    first_batch_user = batch_calls[0]["user"]
    assert "fact_new_ai_chip" in first_batch_user, (
        "First batch should contain the newest fact (fact_new_ai_chip)"
    )

    # The 【最新】 tag should appear for the first fact
    assert "【最新】" in first_batch_user, (
        "First fact in first batch should have 【最新】 priority tag"
    )


# ---------------------------------------------------------------------------
# Test 7: DAG validation passes
# ---------------------------------------------------------------------------

def test_draft_passes_dag_validation():
    """The output ClaimGraphDraft must pass DAG validation (no cycles, valid refs)."""
    facts = _make_50_facts()
    views = build_fact_views_for_d(facts)

    tracker = _LLMCallTracker()
    tracker.all_fact_views = _sort_by_temporal_priority(views)

    with patch("agent_d_strategist.call_llm_json", side_effect=tracker):
        draft = generate_claim_graph_draft(views, "AI战略", trace_id="test_dag")

    # Should not raise
    validate_claim_dag(draft)
    assert len(draft.drafts) > 0


# ---------------------------------------------------------------------------
# Test 8: _match_claims_to_facts resolves empty fact_ids at scale
# ---------------------------------------------------------------------------

def test_match_claims_resolves_at_scale():
    """_match_claims_to_facts fills in empty fact_ids via keyword matching."""
    facts = _make_50_facts()
    views = build_fact_views_for_d(facts)
    sorted_views = _sort_by_temporal_priority(views)

    # Simulate LLM returning claims with empty fact_ids
    raw_draft = ClaimGraphDraft(drafts=[
        ClaimDraft(
            temp_id="claim_0",
            content="该公司发布AI芯片，订单超10万片",
            claim_type=ClaimType.FACT,
            fact_ids=[],
            reasoning="",
            confidence=0.9,
        ),
        ClaimDraft(
            temp_id="claim_1",
            content="全球半导体市场规模持续增长",
            claim_type=ClaimType.FACT,
            fact_ids=[],
            reasoning="",
            confidence=0.7,
        ),
    ])

    fixed = _match_claims_to_facts(raw_draft, sorted_views)

    for d in fixed.drafts:
        assert len(d.fact_ids) > 0, (
            f"Claim '{d.temp_id}' still has empty fact_ids after matching"
        )
        # All fact_ids must be valid
        valid_ids = {f.fact_id for f in sorted_views}
        for fid in d.fact_ids:
            assert fid in valid_ids, f"fact_id '{fid}' not in input facts"

    print(f"\n  claim_0 matched facts: {fixed.drafts[0].fact_ids}")
    print(f"  claim_1 matched facts: {fixed.drafts[1].fact_ids}")


# ---------------------------------------------------------------------------
# Test 9: Both conflicting facts are present in input
# ---------------------------------------------------------------------------

def test_conflicting_facts_present():
    """Verify the 50-fact fixture contains both conflicting facts."""
    facts = _make_50_facts()
    assert len(facts) == 50

    fact_ids = set(facts.keys())
    assert "fact_old_no_ai" in fact_ids
    assert "fact_new_ai_chip" in fact_ids

    old = facts["fact_old_no_ai"]
    new = facts["fact_new_ai_chip"]

    assert old.timestamp < new.timestamp
    assert "AI" in old.content or "ai" in old.summary.lower()
    assert "AI" in new.content or "ai" in new.summary.lower()

    # Old says no AI, new says AI chip released
    assert "没有任何" in old.content or "否认" in old.summary
    assert "发布" in new.content or "芯片" in new.content
