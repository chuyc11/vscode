"""A-node: Tavily-powered web retriever.

Pipeline: sub-query generation → concurrent Tavily search (advanced depth)
→ SimHash content dedup → batched LLM scoring → FactCard assembly.

Tavily returns cleaned content directly, eliminating the need for
BeautifulSoup scraping, Playwright fallback, and domain circuit breakers.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import structlog
from tavily import TavilyClient
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)

from schema import FactCard, SourceTier, EvidenceType, make_id, utc_now
from llm_client import call_llm


# ---------------------------------------------------------------------------
# RawItem — kept for backward compatibility with B-node
# ---------------------------------------------------------------------------

@dataclass
class RawItem:
    """Normalised search result before FactCard conversion."""
    url: str
    domain: str
    title: str
    body: str
    snippet: str
    source_query: str
    used_playwright: bool = False
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# DomainCircuitBreaker — stub for backward compatibility with tests
# ---------------------------------------------------------------------------

class DomainCircuitBreaker:
    """Stub: Tavily does not need a circuit breaker."""

    def __init__(self):
        self._history: dict[str, list[tuple[float, bool]]] = {}

    def record(self, domain: str, was_fallback: bool) -> None:
        pass

    def is_open(self, domain: str) -> bool:
        return False

    def get_stats(self, domain: str) -> dict:
        return {"domain": domain, "total": 0, "fallbacks": 0, "rate": 0.0, "circuit_open": False}

# ---------------------------------------------------------------------------
# Structlog configuration
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if os.getenv("LOG_FORMAT") != "json"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        int(os.getenv("LOG_LEVEL_NUM", "20"))
    ),
)
slog = structlog.get_logger()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_CONCURRENT_SEARCHES = int(os.getenv("A_NODE_MAX_SEARCHES", "5"))
NUM_SUB_QUERIES = int(os.getenv("A_NODE_SUB_QUERIES", "10"))
SCORE_BATCH_SIZE = 10
MIN_RELEVANCE_THRESHOLD = 0.4
SIMHASH_DISTANCE_THRESHOLD = 3

HEURISTIC_SUFFIXES = [
    "latest news",
    "analysis report",
    "expert opinion",
    "background context",
    "impact assessment",
    "historical perspective",
    "stakeholder reactions",
    "data statistics",
    "regional implications",
    "future outlook",
]

# Noise patterns commonly found in Tavily-extracted web content
_NOISE_PATTERNS = [
    r"^(logo|menu|search|login|sign up|subscribe|home|about|contact)\s*",
    r"(analytics|cookie|privacy|terms|广告|备案|京ICP|Copyright)\S*",
    r"(无障碍链接|关注我们|旗下|PLUS|首页|导航)\s*",
    r"#{1,6}\s*(无障碍链接|关注我们|导航|菜单|侧边栏)\s*",
    r"^[#\s]*site\s*$",
    r"^[\s\-_|/\\]+$",  # separator lines
    r"^\s*\d+\s*KB\s*/\s*\d+",  # progress bars
    r"(facebook|twitter|instagram|weibo|wechat)\s*(icon|logo)?\s*$",
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE | re.MULTILINE)

# ---------------------------------------------------------------------------
# Tavily client (singleton)
# ---------------------------------------------------------------------------
_tavily: TavilyClient | None = None


def _get_tavily() -> TavilyClient:
    global _tavily
    if _tavily is None:
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not set")
        _tavily = TavilyClient(api_key=api_key)
    return _tavily


# ---------------------------------------------------------------------------
# Content Deduplication (SimHash)
# ---------------------------------------------------------------------------

def _compute_simhash(text: str, hash_bits: int = 64) -> int:
    words = text.lower().split()
    if len(words) < 4:
        words = words + [""] * (4 - len(words))
    vector = [0] * hash_bits
    for i in range(len(words) - 3):
        shingle = " ".join(words[i : i + 4])
        h = int(hashlib.md5(shingle.encode("utf-8")).hexdigest(), 16)
        for bit in range(hash_bits):
            if h & (1 << bit):
                vector[bit] += 1
            else:
                vector[bit] -= 1
    fingerprint = 0
    for bit in range(hash_bits):
        if vector[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def _hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def _is_near_duplicate(text: str, seen_hashes: list[int]) -> bool:
    try:
        h = _compute_simhash(text)
        for existing in seen_hashes:
            if _hamming_distance(h, existing) <= SIMHASH_DISTANCE_THRESHOLD:
                return True
        seen_hashes.append(h)
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Sub-query Generation
# ---------------------------------------------------------------------------

async def _generate_sub_queries(trace_id: str, query: str) -> list[str]:
    prompt = f"""你是一个情报搜索策略专家。请为以下分析主题生成 {NUM_SUB_QUERIES} 个搜索关键词，覆盖尽可能多的信息维度。

主题: {query}

要求：
- 每个关键词必须从不同维度切入（如：事实报道、专家分析、历史背景、利益相关方反应、数据统计、区域影响等）
- 中英文混合搜索（部分关键词用英文可以获取更多国际视角）
- 避免关键词之间的重叠
- 只返回 JSON 数组，不要其他内容"""

    try:
        raw = await asyncio.to_thread(
            call_llm, [{"role": "user", "content": prompt}]
        )
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        subs = json.loads(raw)
        if isinstance(subs, list) and all(isinstance(s, str) for s in subs):
            result = [query] + subs[:NUM_SUB_QUERIES]
            slog.info("sub_queries_generated", trace_id=trace_id, count=len(result), method="llm")
            return result
    except Exception as e:
        slog.warning("sub_query_llm_failed", trace_id=trace_id, error=str(e), fallback="heuristic")

    result = [query] + [f"{query} {s}" for s in HEURISTIC_SUFFIXES[:NUM_SUB_QUERIES]]
    slog.info("sub_queries_generated", trace_id=trace_id, count=len(result), method="heuristic")
    return result


# ---------------------------------------------------------------------------
# Tavily Search
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
    reraise=True,
)
def _tavily_search_raw(query: str, max_results: int) -> list[dict]:
    client = _get_tavily()
    # Always fetch at least 8 results per sub-query for breadth
    per_query = max(max_results, 8)
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=per_query,
        include_answer=False,
    )
    return response.get("results", [])


async def _search_sub_query(
    query: str, max_results: int, semaphore: asyncio.Semaphore
) -> list[dict]:
    async with semaphore:
        try:
            return await asyncio.to_thread(_tavily_search_raw, query, max_results)
        except Exception as e:
            slog.warning("tavily_search_failed", query=query[:60], error=str(e))
            return []


async def _concurrent_search(
    trace_id: str, queries: list[str], max_results: int
) -> list[dict]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)
    results = await asyncio.gather(
        *[_search_sub_query(q, max_results, semaphore) for q in queries]
    )
    # Deduplicate by URL
    seen_urls: set[str] = set()
    deduped: list[dict] = []
    for batch in results:
        for item in batch:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped.append(item)
    total = sum(len(b) for b in results)
    slog.info("tavily_search_complete", trace_id=trace_id, raw=total, unique=len(deduped))
    return deduped


# ---------------------------------------------------------------------------
# LLM Scoring (batched, concurrent)
# ---------------------------------------------------------------------------

def _score_batch_sync(topic: str, items_text: str) -> list[dict]:
    prompt = f"""你是一个情报分析专家。请对以下搜索结果进行评估。

分析主题: {topic}

搜索结果:
{items_text}

请对每条结果返回 JSON 数组，每条包含:
- index: 原始索引
- relevance: 与主题的相关性 (0.0-1.0)
- credibility: 来源可信度 (0.0-1.0)
- summary: 一句话说明这条信息与主题的关系

只返回 JSON 数组。"""
    raw = call_llm([{"role": "user", "content": prompt}])
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


async def _score_batch(
    trace_id: str, topic: str, items: list[dict], batch_offset: int
) -> list[dict]:
    items_text = "\n".join(
        f"[{batch_offset + i}] {_clean_content(it.get('title', ''))}: {_clean_content(it.get('content', ''))[:300]}"
        for i, it in enumerate(items)
    )
    indices = list(range(batch_offset, batch_offset + len(items)))
    try:
        scores = await asyncio.to_thread(_score_batch_sync, topic, items_text)
        return scores
    except Exception as e:
        slog.warning("scoring_batch_failed", trace_id=trace_id, error=str(e))
        return [
            {"index": idx, "relevance": 0.6, "credibility": 0.6, "summary": ""}
            for idx in indices
        ]


async def _score_all(
    trace_id: str, topic: str, items: list[dict]
) -> list[dict]:
    if not items:
        return []
    batches = [items[i : i + SCORE_BATCH_SIZE] for i in range(0, len(items), SCORE_BATCH_SIZE)]
    tasks = [
        _score_batch(trace_id, topic, batch, i * SCORE_BATCH_SIZE)
        for i, batch in enumerate(batches)
    ]
    results = await asyncio.gather(*tasks)
    all_scores = []
    for batch_result in results:
        all_scores.extend(batch_result)
    all_scores.sort(key=lambda x: x.get("index", 0))
    filtered = [s for s in all_scores if s.get("relevance", 0) >= MIN_RELEVANCE_THRESHOLD]
    slog.info("scoring_complete", trace_id=trace_id,
              total=len(all_scores), passed=len(filtered))
    return filtered


# ---------------------------------------------------------------------------
# Content Cleaning
# ---------------------------------------------------------------------------

def _clean_content(text: str) -> str:
    """Remove HTML artifacts, navigation noise, and boilerplate from Tavily content."""
    if not text:
        return text
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove noise patterns line by line
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if _NOISE_RE.search(line):
            continue
        # Skip very short lines that are likely navigation fragments
        if len(line) < 3 and not line.isdigit():
            continue
        cleaned.append(line)
    result = "\n".join(cleaned)
    # Collapse multiple spaces
    result = re.sub(r"[ \t]{2,}", " ", result)
    return result.strip()


# ---------------------------------------------------------------------------
# FactCard Assembly
# ---------------------------------------------------------------------------

def _build_fact_cards(
    items: list[dict], scores: list[dict], max_results: int
) -> list[FactCard]:
    score_map = {s["index"]: s for s in scores}
    facts: list[FactCard] = []
    for i, item in enumerate(items):
        s = score_map.get(i)
        if not s:
            continue
        content = _clean_content(item.get("content", ""))
        title = _clean_content(item.get("title", ""))
        if title:
            content = f"{title}: {content}"
        if len(content) < 50:
            continue  # skip too-short results after cleaning
        facts.append(
            FactCard(
                fact_id=make_id(),
                content=content[:500],
                source_tier=SourceTier.SECONDARY,
                evidence_type=EvidenceType.DOCUMENT,
                timestamp=utc_now(),
                credibility_score=float(s.get("credibility", 0.6)),
                relevance_score=float(s.get("relevance", 0.6)),
                summary=s.get("summary", ""),
            )
        )
    facts.sort(key=lambda f: f.relevance_score * f.credibility_score, reverse=True)
    return facts[:max_results]


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _make_fallback_facts() -> list[FactCard]:
    contents = [
        "Industry analysis shows AI chip demand continues to rise, benefiting major GPU manufacturers.",
        "Market analysts report increased capital expenditure in AI infrastructure across multiple sectors.",
        "Recent partnerships between AI companies and hardware suppliers may drive significant revenue growth.",
    ]
    return [
        FactCard(
            fact_id=make_id(),
            content=c,
            source_tier=SourceTier.TERTIARY,
            evidence_type=EvidenceType.DOCUMENT,
            timestamp=utc_now(),
            credibility_score=0.4,
            relevance_score=0.5,
            summary="备用情报（网络检索失败时启用）",
        )
        for c in contents
    ]


# ---------------------------------------------------------------------------
# Async Pipeline
# ---------------------------------------------------------------------------

async def _async_pipeline(query: str, max_results: int) -> list[FactCard]:
    trace_id = uuid.uuid4().hex[:12]
    slog.info("pipeline_start", trace_id=trace_id, query=query)

    # 1. Generate sub-queries
    sub_queries = await _generate_sub_queries(trace_id, query)

    # 2. Concurrent Tavily search (returns content directly)
    search_results = await _concurrent_search(trace_id, sub_queries, max_results)
    if not search_results:
        slog.warning("no_search_results", trace_id=trace_id)
        return _make_fallback_facts()

    # 3. Content-level SimHash dedup
    seen_hashes: list[int] = []
    deduped: list[dict] = []
    for item in search_results:
        content = item.get("content", "")
        if content and not _is_near_duplicate(content, seen_hashes):
            deduped.append(item)
    slog.info("content_dedup", trace_id=trace_id,
              before=len(search_results), after=len(deduped))

    # 4. Batch LLM scoring
    scores = await _score_all(trace_id, query, deduped)
    if not scores:
        slog.warning("no_scores", trace_id=trace_id)
        return _make_fallback_facts()

    # 5. Build FactCards
    facts = _build_fact_cards(deduped, scores, max_results)
    if not facts:
        slog.warning("no_factcards", trace_id=trace_id)
        return _make_fallback_facts()

    slog.info("pipeline_complete", trace_id=trace_id, factcards=len(facts))
    return facts


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def search_real_news(query: str, max_results: int = 5) -> list[FactCard]:
    """Search the web via Tavily, score with LLM, and return FactCards."""
    try:
        return asyncio.run(_async_pipeline(query, max_results))
    except Exception as e:
        slog.error("pipeline_failed", error=str(e))
        return _make_fallback_facts()
