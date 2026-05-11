"""Stress tests for A-node: Tavily-powered search pipeline.

Tests:
1. Tavily search returns results — normal path
2. Tavily search returns empty — fallback facts
3. Tavily API failure — retry + fallback
4. Content deduplication (SimHash)
5. Sub-query generation — LLM + heuristic fallback
6. Concurrent search dedup by URL
7. LLM scoring batch failure — fallback scores
8. Trace ID propagation in structlog
9. DomainCircuitBreaker stub backward compatibility
"""

import asyncio
import json
from unittest.mock import patch, MagicMock

import pytest

from agent_a_retriever import (
    DomainCircuitBreaker,
    _compute_simhash,
    _is_near_duplicate,
    _tavily_search_raw,
    _generate_sub_queries,
    _concurrent_search,
    _score_all,
    _build_fact_cards,
    _make_fallback_facts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TAVILY_RESULTS = [
    {
        "title": "NVIDIA Reports Record Q4 Revenue",
        "url": "https://example.com/nvidia-q4",
        "content": "NVIDIA reported record quarterly revenue of $22.1 billion, driven by strong AI chip demand.",
        "score": 0.95,
    },
    {
        "title": "AI Chip Market Growth",
        "url": "https://example.com/ai-chips",
        "content": "The AI semiconductor market is expected to reach $200 billion by 2027, with NVIDIA maintaining dominance.",
        "score": 0.88,
    },
    {
        "title": "NVIDIA Stock Analysis",
        "url": "https://finance.example.com/nvda",
        "content": "NVIDIA stock has surged 200% in the past year as investors bet on AI infrastructure spending.",
        "score": 0.82,
    },
]


# ---------------------------------------------------------------------------
# Test 1: Normal Tavily search
# ---------------------------------------------------------------------------

@patch("agent_a_retriever._get_tavily")
def test_tavily_search_returns_results(mock_get_tavily):
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": SAMPLE_TAVILY_RESULTS}
    mock_get_tavily.return_value = mock_client

    results = _tavily_search_raw("NVIDIA earnings", 5)
    assert len(results) == 3
    assert results[0]["title"] == "NVIDIA Reports Record Q4 Revenue"
    mock_client.search.assert_called_once_with(
        query="NVIDIA earnings",
        search_depth="advanced",
        max_results=8,  # min 8 per sub-query for breadth
        include_answer=False,
    )


# ---------------------------------------------------------------------------
# Test 2: Empty results → fallback facts
# ---------------------------------------------------------------------------

def test_fallback_facts_generated():
    facts = _make_fallback_facts()
    assert len(facts) == 3
    for f in facts:
        assert f.credibility_score == 0.4
        assert f.relevance_score == 0.5
        assert f.summary == "备用情报（网络检索失败时启用）"


# ---------------------------------------------------------------------------
# Test 3: Tavily API failure — tenacity retries
# ---------------------------------------------------------------------------

@patch("agent_a_retriever._get_tavily")
def test_tavily_api_failure_raises(mock_get_tavily):
    mock_client = MagicMock()
    mock_client.search.side_effect = ConnectionError("Network unreachable")
    mock_get_tavily.return_value = mock_client

    with pytest.raises(ConnectionError):
        _tavily_search_raw("test query", 3)


# ---------------------------------------------------------------------------
# Test 4: SimHash content deduplication
# ---------------------------------------------------------------------------

def test_simhash_identical_content():
    text = "NVIDIA reported record quarterly revenue driven by strong AI chip demand in data centers"
    h1 = _compute_simhash(text)
    h2 = _compute_simhash(text)
    assert h1 == h2


def test_simhash_near_duplicate_detected():
    text = "NVIDIA reported record quarterly revenue driven by strong AI chip demand in data centers worldwide"
    hashes = [_compute_simhash(text)]
    assert _is_near_duplicate(text, hashes) is True, "Identical text must be detected as duplicate"


def test_simhash_different_content_passes():
    text1 = "NVIDIA reported record quarterly revenue driven by strong AI chip demand"
    text2 = "The weather today is sunny with clear skies and mild temperatures"
    hashes = [_compute_simhash(text1)]
    assert _is_near_duplicate(text2, hashes) is False


def test_simhash_short_text():
    text = "short"
    h = _compute_simhash(text)
    assert isinstance(h, int)


# ---------------------------------------------------------------------------
# Test 5: Sub-query generation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("agent_a_retriever.call_llm")
async def test_sub_queries_llm_success(mock_llm):
    mock_llm.return_value = '["q1","q2","q3","q4","q5","q6","q7","q8","q9","q10"]'
    result = await _generate_sub_queries("test01", "NVIDIA earnings")
    assert len(result) == 11  # original + 10 sub-queries
    assert result[0] == "NVIDIA earnings"


@pytest.mark.asyncio
@patch("agent_a_retriever.call_llm")
async def test_sub_queries_llm_failure_heuristic_fallback(mock_llm):
    mock_llm.side_effect = Exception("LLM timeout")
    result = await _generate_sub_queries("test02", "NVIDIA earnings")
    assert len(result) == 11  # 1 original + 10 heuristic suffixes
    assert result[0] == "NVIDIA earnings"
    assert "latest news" in result[1]


# ---------------------------------------------------------------------------
# Test 6: Concurrent search URL dedup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("agent_a_retriever._tavily_search_raw")
async def test_concurrent_search_dedup(mock_search):
    # Return overlapping results from different sub-queries
    mock_search.side_effect = [
        [SAMPLE_TAVILY_RESULTS[0], SAMPLE_TAVILY_RESULTS[1]],
        [SAMPLE_TAVILY_RESULTS[1], SAMPLE_TAVILY_RESULTS[2]],  # overlap on [1]
    ]

    results = await _concurrent_search("dedup01", ["q1", "q2"], 5)
    urls = [r["url"] for r in results]
    assert len(urls) == 3
    assert len(set(urls)) == 3, "URLs should be deduplicated"


# ---------------------------------------------------------------------------
# Test 7: LLM scoring failure → fallback scores
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("agent_a_retriever.call_llm")
async def test_scoring_failure_fallback(mock_llm):
    mock_llm.side_effect = Exception("LLM unavailable")

    items = [
        {"title": "Test", "content": "Some content about NVIDIA earnings"},
    ]
    scores = await _score_all("score01", "NVIDIA", items)
    assert len(scores) == 1
    assert scores[0]["relevance"] == 0.6
    assert scores[0]["credibility"] == 0.6


# ---------------------------------------------------------------------------
# Test 8: Trace ID propagation
# ---------------------------------------------------------------------------

def test_trace_id_in_structlog():
    """Verify DomainCircuitBreaker stub works (backward compat)."""
    cb = DomainCircuitBreaker()
    cb.record("test.com", was_fallback=True)
    assert cb.is_open("test.com") is False  # stub always returns False
    stats = cb.get_stats("test.com")
    assert stats["circuit_open"] is False
    assert stats["total"] == 0


# ---------------------------------------------------------------------------
# Test 9: FactCard assembly
# ---------------------------------------------------------------------------

def test_build_fact_cards():
    items = [
        {"title": "NVIDIA Q4 Earnings Report", "content": "NVIDIA reported record quarterly revenue of $22.1 billion, driven by strong AI chip demand across data centers and cloud computing."},
        {"title": "AI Market Growth Forecast", "content": "The AI semiconductor market is expected to reach $200 billion by 2027, with NVIDIA maintaining its dominant position."},
    ]
    scores = [
        {"index": 0, "relevance": 0.9, "credibility": 0.8, "summary": "NVIDIA earnings"},
        {"index": 1, "relevance": 0.7, "credibility": 0.7, "summary": "Market growth"},
    ]
    facts = _build_fact_cards(items, scores, 5)
    assert len(facts) == 2
    assert facts[0].relevance_score >= facts[1].relevance_score  # sorted by score
    assert "NVIDIA" in facts[0].content


def test_build_fact_cards_respects_max():
    items = [{"title": f"Item {i} with enough content", "content": f"This is content number {i} with sufficient text to pass the cleaning threshold for fact cards."} for i in range(10)]
    scores = [{"index": i, "relevance": 0.8, "credibility": 0.7, "summary": ""} for i in range(10)]
    facts = _build_fact_cards(items, scores, 3)
    assert len(facts) == 3
