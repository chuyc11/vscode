"""Stress tests for C-node context-aware ticker alignment.

Feeds ambiguous entity names (Apple, Ford, Amazon) with and without
financial context. Verifies the context-aware validation:
- Rejects blind ticker alignment when no financial context provided
- Logs structlog degradation events (entity_mapping_rejected, c_node_no_valid_mappings)
- Does NOT trigger OpenBB fetches for rejected entities
- Accepts same entities when proper financial context exists

Engineering red lines verified:
- context_aligned=False → entity rejected regardless of confidence
- Confidence below MAPPING_CONFIDENCE_THRESHOLD → rejected
- LLM exception → pipeline degrades gracefully, no crash
- Zero OpenBB calls when all entities rejected
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from agent_c_financial import (
    enrich_financial_data,
    enrich_financial_data_sync,
    MAPPING_CONFIDENCE_THRESHOLD,
)
from schema import (
    Entity, FactCard, SourceTier, EvidenceType, make_id, utc_now,
    PriceData, FundamentalData,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fact(entity_text: str, content: str = "", summary: str = "") -> FactCard:
    """Construct a FactCard with a single ORG entity."""
    return FactCard(
        fact_id=make_id(),
        content=content,
        summary=summary,
        source_tier=SourceTier.SECONDARY,
        evidence_type=EvidenceType.DOCUMENT,
        entities=[Entity(text=entity_text, label="ORG", start=0, end=len(entity_text))],
    )


def _make_facts(entities: dict[str, tuple[str, str]]) -> dict[str, FactCard]:
    """Build FactCard dict from {entity_text: (content, summary)} mapping."""
    facts = {}
    for entity_text, (content, summary) in entities.items():
        fact = _make_fact(entity_text, content, summary)
        facts[fact.fact_id] = fact
    return facts


TICKER_MAP = {"Apple": "AAPL", "Ford": "F", "Amazon": "AMZN"}


def _mock_llm_no_context(messages, schema):
    """LLM returns context_aligned=False — no financial context provided."""
    entity_text = messages[1]["content"].split("\n")[0].replace("Entity: ", "")
    return json.dumps({
        "ticker": TICKER_MAP.get(entity_text),
        "company_name": entity_text,
        "confidence": 0.3,
        "context_aligned": False,
        "reason": f"No financial context provided for '{entity_text}'",
    })


def _mock_llm_with_context(messages, schema):
    """LLM returns context_aligned=True — financial context confirmed."""
    entity_text = messages[1]["content"].split("\n")[0].replace("Entity: ", "")
    return json.dumps({
        "ticker": TICKER_MAP.get(entity_text),
        "company_name": f"{entity_text} Inc.",
        "confidence": 0.95,
        "context_aligned": True,
        "reason": f"Entity refers to {entity_text} Inc. in financial context",
    })


def _mock_llm_low_confidence(messages, schema):
    """LLM returns context_aligned=True but confidence below threshold."""
    entity_text = messages[1]["content"].split("\n")[0].replace("Entity: ", "")
    return json.dumps({
        "ticker": TICKER_MAP.get(entity_text),
        "company_name": f"{entity_text} Inc.",
        "confidence": 0.4,
        "context_aligned": True,
        "reason": f"Possible match for {entity_text} but low confidence",
    })


def _make_mock_llm_mixed(accepted_entities: set[str]):
    """Factory: returns mock that accepts specified entities, rejects others."""
    def _mock(messages, schema):
        entity_text = messages[1]["content"].split("\n")[0].replace("Entity: ", "")
        accepted = entity_text in accepted_entities
        return json.dumps({
            "ticker": TICKER_MAP.get(entity_text),
            "company_name": f"{entity_text} Inc.",
            "confidence": 0.95 if accepted else 0.3,
            "context_aligned": accepted,
            "reason": f"{'Financial context found' if accepted else 'No financial context'} for {entity_text}",
        })
    return _mock


def _mock_openbb_factory():
    """Create mock OpenBB module with price/fundamentals returning test data."""

    class MockResults:
        def __init__(self, data):
            self.results = data

    class MockPriceQuote:
        last_price = 150.0
        prev_close = 148.0
        open_price = 149.0
        day_high = 152.0
        day_low = 147.0
        volume = 50000000
        market_cap = 2500000000000
        fifty_two_week_high = 199.0
        fifty_two_week_low = 124.0
        currency = "USD"

    class MockFundamental:
        price_earnings_ratio = 28.5
        price_to_book_ratio = 45.0
        price_to_sales_ratio = 7.5
        dividend_yield = 0.005
        revenue_per_share_ttm = 24.3
        net_income_per_share_ttm = 6.1
        earnings_per_share_ttm = 6.1
        return_on_equity = 1.71
        debt_to_equity_ratio = 1.8
        current_ratio = 1.0
        gross_margin = 0.45
        operating_margin = 0.30
        beta = 1.2

    mock_obb = MagicMock()
    mock_obb.equity.price.quote.return_value = MockResults([MockPriceQuote()])
    mock_obb.equity.fundamental.ratios.return_value = MockResults([MockFundamental()])
    return mock_obb


# ---------------------------------------------------------------------------
# Test 1: Core — ambiguous entities without context are rejected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ambiguous_entities_no_context_rejected():
    """Apple, Ford, Amazon with empty content/summary → all rejected."""
    facts = _make_facts({
        "Apple": ("", ""),
        "Ford": ("", ""),
        "Amazon": ("", ""),
    })

    with patch("agent_c_financial.call_llm_json", side_effect=_mock_llm_no_context):
        result = await enrich_financial_data(facts, trace_id="test_no_ctx")

    assert result == [], "All entities should be rejected without financial context"


# ---------------------------------------------------------------------------
# Test 2: structlog rejection events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_structlog_rejection_events():
    """Verify structlog emits entity_mapping_rejected and c_node_no_valid_mappings."""
    from structlog.testing import LogCapture
    import structlog

    facts = _make_facts({
        "Apple": ("", ""),
        "Ford": ("", ""),
        "Amazon": ("", ""),
    })

    cap = LogCapture()
    old_processors = structlog.get_config()["processors"]

    try:
        structlog.configure(processors=[cap])
        with patch("agent_c_financial.call_llm_json", side_effect=_mock_llm_no_context):
            await enrich_financial_data(facts, trace_id="test_structlog")
    finally:
        structlog.configure(processors=old_processors)

    # Check for rejection events
    rejection_events = [e for e in cap.entries if e.get("event") == "entity_mapping_rejected"]
    assert len(rejection_events) >= 1, (
        f"Expected entity_mapping_rejected events, got {len(rejection_events)}"
    )

    for evt in rejection_events:
        assert evt["context_aligned"] is False
        assert evt["confidence"] < MAPPING_CONFIDENCE_THRESHOLD
        assert "entity" in evt
        assert "reason" in evt

    no_mapping_events = [e for e in cap.entries if e.get("event") == "c_node_no_valid_mappings"]
    assert len(no_mapping_events) == 1, "Should have exactly one c_node_no_valid_mappings event"


# ---------------------------------------------------------------------------
# Test 3: No OpenBB fetch when all entities rejected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_openbb_fetch_when_all_rejected():
    """Zero OpenBB calls when all entities are context-rejected."""
    facts = _make_facts({
        "Apple": ("", ""),
        "Ford": ("", ""),
        "Amazon": ("", ""),
    })

    mock_obb = _mock_openbb_factory()

    with patch("agent_c_financial.call_llm_json", side_effect=_mock_llm_no_context):
        import sys
        sys.modules["openbb"] = MagicMock()
        sys.modules["openbb"].obb = mock_obb
        try:
            result = await enrich_financial_data(facts, trace_id="test_no_openbb")
        finally:
            sys.modules.pop("openbb", None)

    assert result == []
    mock_obb.equity.price.quote.assert_not_called()
    mock_obb.equity.fundamental.ratios.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: Control — same entities WITH context are accepted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ambiguous_entities_with_context_accepted():
    """Same ambiguous names but with financial context → accepted, QuantFactCards returned."""
    facts = _make_facts({
        "Apple": (
            "Apple Inc. reported record quarterly revenue of $94.8 billion, driven by strong iPhone sales.",
            "Apple Q1 2024 earnings report",
        ),
        "Ford": (
            "Ford Motor Company announced a $5 billion investment in electric vehicle production.",
            "Ford EV investment strategy",
        ),
        "Amazon": (
            "Amazon Web Services revenue grew 13% year-over-year, beating analyst expectations.",
            "Amazon AWS quarterly growth",
        ),
    })

    mock_obb = _mock_openbb_factory()

    with patch("agent_c_financial.call_llm_json", side_effect=_mock_llm_with_context):
        import sys
        sys.modules["openbb"] = MagicMock()
        sys.modules["openbb"].obb = mock_obb
        try:
            result = await enrich_financial_data(facts, trace_id="test_with_ctx")
        finally:
            sys.modules.pop("openbb", None)

    assert len(result) == 3, f"Expected 3 QuantFactCards, got {len(result)}"

    tickers = {card.ticker for card in result}
    assert tickers == {"AAPL", "F", "AMZN"}

    for card in result:
        assert card.mapping_confidence == 0.95
        assert card.fetch_success is True
        assert card.price.last_price == 150.0

    # OpenBB should have been called 3 times (once per ticker)
    assert mock_obb.equity.price.quote.call_count == 3


# ---------------------------------------------------------------------------
# Test 5: Mixed — some accepted, some rejected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mixed_accepted_and_rejected():
    """Apple has context (accepted), Amazon has no context (rejected)."""
    facts = _make_facts({
        "Apple": (
            "Apple Inc. quarterly revenue hit $94.8 billion in Q1 2024.",
            "Apple earnings report",
        ),
        "Amazon": ("", ""),  # No context → rejected
    })

    mock_obb = _mock_openbb_factory()

    mock_llm = _make_mock_llm_mixed({"Apple"})  # Only Apple accepted
    with patch("agent_c_financial.call_llm_json", side_effect=mock_llm):
        import sys
        sys.modules["openbb"] = MagicMock()
        sys.modules["openbb"].obb = mock_obb
        try:
            result = await enrich_financial_data(facts, trace_id="test_mixed")
        finally:
            sys.modules.pop("openbb", None)

    assert len(result) == 1, f"Expected 1 QuantFactCard (Apple only), got {len(result)}"
    assert result[0].ticker == "AAPL"

    # Only 1 OpenBB call (for AAPL)
    assert mock_obb.equity.price.quote.call_count == 1


# ---------------------------------------------------------------------------
# Test 6: Low confidence below threshold → rejected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_low_confidence_below_threshold_rejected():
    """context_aligned=True but confidence=0.4 < 0.6 threshold → rejected."""
    facts = _make_facts({
        "Ford": (
            "Ford Motor Company earnings report for Q1 2024.",
            "Ford quarterly results",
        ),
    })

    with patch("agent_c_financial.call_llm_json", side_effect=_mock_llm_low_confidence):
        result = await enrich_financial_data(facts, trace_id="test_low_conf")

    assert result == [], "Low confidence should be rejected even with context_aligned=True"


# ---------------------------------------------------------------------------
# Test 7: LLM exception → graceful degradation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_exception_does_not_crash_pipeline():
    """LLM raises exception → pipeline returns empty list, no crash."""
    from structlog.testing import LogCapture
    import structlog

    facts = _make_facts({
        "Apple": ("", ""),
        "Ford": ("", ""),
        "Amazon": ("", ""),
    })

    cap = LogCapture()
    old_processors = structlog.get_config()["processors"]

    try:
        structlog.configure(processors=[cap])
        with patch("agent_c_financial.call_llm_json", side_effect=Exception("LLM timeout")):
            result = await enrich_financial_data(facts, trace_id="test_llm_exc")
    finally:
        structlog.configure(processors=old_processors)

    assert result == [], "LLM exception should not crash pipeline"

    warning_events = [e for e in cap.entries if e.get("event") in ("mapping_exception", "ticker_mapping_failed")]
    assert len(warning_events) >= 1, (
        f"Expected mapping_exception or ticker_mapping_failed warning, got events: "
        f"{[e.get('event') for e in cap.entries]}"
    )
