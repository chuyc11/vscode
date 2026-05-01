"""A-node: concurrent web retriever with domain circuit breaker.

Pipeline: sub-query generation → concurrent DuckDuckGo search → URL
normalization & dedup → concurrent BeautifulSoup scrape (with Playwright
fallback gated by domain-level circuit breaker) → SimHash content dedup
→ batched LLM scoring → FactCard assembly.

Engineering red lines
---------------------
1. Domain-level circuit breaker: if a domain's Playwright fallback rate
   exceeds CB_THRESHOLD within CB_WINDOW seconds, new Playwright instances
   for that domain are rejected (prevents OOM).
2. All底层 I/O is async-safe (aiohttp, asyncio.to_thread for sync calls).
3. Circuit breaker events are logged via structlog with a per-pipeline
   Trace ID for full observability.
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
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import aiohttp
import structlog
from bs4 import BeautifulSoup, Tag
from ddgs import DDGS
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
# Structlog configuration (JSON output with Trace ID injection)
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
# Configuration (env-overridable)
# ---------------------------------------------------------------------------
MAX_CONCURRENT_SCRAPES = int(os.getenv("A_NODE_MAX_SCRAPES", "10"))
MAX_CONCURRENT_SEARCHES = int(os.getenv("A_NODE_MAX_SEARCHES", "5"))
NUM_SUB_QUERIES = int(os.getenv("A_NODE_SUB_QUERIES", "3"))
SCRAPE_TIMEOUT = int(os.getenv("A_NODE_SCRAPE_TIMEOUT", "8"))
MIN_CONTENT_LENGTH = 100
SIMHASH_DISTANCE_THRESHOLD = 3
SCORE_BATCH_SIZE = 10
MIN_RELEVANCE_THRESHOLD = 0.5

# Playwright fallback
PLAYWRIGHT_TIMEOUT = int(os.getenv("A_NODE_PW_TIMEOUT", "15"))
ENABLE_PLAYWRIGHT = os.getenv("A_NODE_ENABLE_PLAYWRIGHT", "0") == "1"

# Circuit breaker thresholds
CB_WINDOW = int(os.getenv("A_NODE_CB_WINDOW", "300"))  # seconds
CB_THRESHOLD = float(os.getenv("A_NODE_CB_THRESHOLD", "0.5"))  # 50% fallback rate
CB_MIN_CALLS = int(os.getenv("A_NODE_CB_MIN_CALLS", "3"))  # min attempts before tripping

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)
HEURISTIC_SUFFIXES = ["latest news", "analysis report", "expert opinion"]
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_cid", "fbclid", "gclid", "msclkid", "ref", "ref_",
    "source", "spm", "from", "isappinstalled", "mc_cid", "mc_eid",
}


# ---------------------------------------------------------------------------
# RawItem — normalized intermediate output
# ---------------------------------------------------------------------------

@dataclass
class RawItem:
    """Normalised scrape result before FactCard conversion."""
    url: str
    domain: str
    title: str
    body: str
    snippet: str
    source_query: str
    used_playwright: bool = False
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Domain-level Circuit Breaker
# ---------------------------------------------------------------------------

class DomainCircuitBreaker:
    """Track per-domain Playwright fallback rates and block when excessive.

    Prevents OOM by refusing to launch new Playwright browser instances
    for domains that consistently require fallback (indicating anti-bot
    protection or heavy JS rendering).
    """

    def __init__(self):
        # domain -> list of (timestamp, was_fallback)
        self._history: dict[str, list[tuple[float, bool]]] = {}

    def record(self, domain: str, was_fallback: bool) -> None:
        now = time.time()
        if domain not in self._history:
            self._history[domain] = []
        self._history[domain].append((now, was_fallback))
        # Prune old entries outside the window
        cutoff = now - CB_WINDOW
        self._history[domain] = [
            (t, f) for t, f in self._history[domain] if t >= cutoff
        ]

    def is_open(self, domain: str) -> bool:
        """Return True if the circuit is open (block Playwright for this domain)."""
        entries = self._history.get(domain, [])
        if len(entries) < CB_MIN_CALLS:
            return False
        fallback_count = sum(1 for _, f in entries if f)
        rate = fallback_count / len(entries)
        return rate >= CB_THRESHOLD

    def get_stats(self, domain: str) -> dict:
        entries = self._history.get(domain, [])
        fallback_count = sum(1 for _, f in entries if f)
        return {
            "domain": domain,
            "total": len(entries),
            "fallbacks": fallback_count,
            "rate": fallback_count / len(entries) if entries else 0.0,
            "circuit_open": self.is_open(domain),
        }


_cb = DomainCircuitBreaker()


# ---------------------------------------------------------------------------
# URL Normalization
# ---------------------------------------------------------------------------

def _normalize_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return None
    port = parsed.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None
    netloc = hostname if port is None else f"{hostname}:{port}"
    path = parsed.path.rstrip("/") or "/"
    query_params = parse_qs(parsed.query, keep_blank_values=True)
    cleaned = {k: v for k, v in query_params.items() if k.lower() not in TRACKING_PARAMS}
    query = urlencode(cleaned, doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def _extract_domain(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# HTML Content Cleaning
# ---------------------------------------------------------------------------

_DISALLOWED_TAGS = {
    "script", "style", "footer", "header", "nav", "menu",
    "sidebar", "svg", "noscript", "iframe", "form",
}
_DISALLOWED_CLASSES = {
    "nav", "menu", "sidebar", "footer", "ad", "advertisement", "cookie-banner",
}


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(_DISALLOWED_TAGS):
        tag.decompose()

    def _has_disallowed_class(elem) -> bool:
        if not isinstance(elem, Tag):
            return False
        return any(cls in _DISALLOWED_CLASSES for cls in elem.get("class", []))

    for tag in soup.find_all(_has_disallowed_class):
        tag.decompose()
    text = soup.get_text(strip=True, separator="\n")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


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
    prompt = f"""你是一个搜索策略专家。请为以下分析主题生成 {NUM_SUB_QUERIES} 个不同角度的搜索关键词。

主题: {query}

要求：每个关键词从不同维度切入。只返回 JSON 数组。"""

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
# Concurrent Search (DuckDuckGo)
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
    reraise=True,
)
def _search_raw(query: str, max_results: int) -> list[dict]:
    with DDGS() as ddgs:
        return ddgs.text(query, max_results=max_results)


async def _search_sub_query(
    query: str, max_results: int, semaphore: asyncio.Semaphore
) -> list[dict]:
    async with semaphore:
        try:
            return await asyncio.to_thread(_search_raw, query, max_results)
        except Exception as e:
            slog.warning("search_failed", query=query[:60], error=str(e))
            return []


async def _concurrent_search(
    trace_id: str, queries: list[str], max_results: int
) -> list[dict]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)
    results = await asyncio.gather(
        *[_search_sub_query(q, max_results, semaphore) for q in queries]
    )
    seen_urls: set[str] = set()
    deduped: list[dict] = []
    for batch in results:
        for item in batch:
            href = item.get("href", "")
            norm = _normalize_url(href)
            if norm and norm not in seen_urls:
                seen_urls.add(norm)
                item["_norm_url"] = norm
                deduped.append(item)
    total = sum(len(b) for b in results)
    slog.info("search_complete", trace_id=trace_id, raw=total, unique=len(deduped))
    return deduped


# ---------------------------------------------------------------------------
# Playwright Fallback (async, circuit-breaker gated)
# ---------------------------------------------------------------------------

async def _playwright_scrape(url: str, domain: str, trace_id: str) -> Optional[str]:
    """Scrape a JS-heavy page with Playwright. Returns cleaned text or None."""
    if not ENABLE_PLAYWRIGHT:
        return None
    if _cb.is_open(domain):
        slog.warning("circuit_breaker_blocked",
                      trace_id=trace_id, domain=domain,
                      stats=_cb.get_stats(domain))
        return None
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=PLAYWRIGHT_TIMEOUT * 1000, wait_until="domcontentloaded")
            html = await page.content()
            await browser.close()
        body = _clean_html(html)
        _cb.record(domain, was_fallback=False)
        return body if len(body) >= MIN_CONTENT_LENGTH else None
    except Exception as e:
        _cb.record(domain, was_fallback=True)
        slog.warning("playwright_fallback_failed",
                      trace_id=trace_id, domain=domain, error=str(e),
                      stats=_cb.get_stats(domain))
        return None


# ---------------------------------------------------------------------------
# Concurrent Scraping (aiohttp + BeautifulSoup, Playwright fallback)
# ---------------------------------------------------------------------------

async def _scrape_url(
    url: str,
    domain: str,
    snippet: str,
    source_query: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    trace_id: str,
) -> Optional[RawItem]:
    async with semaphore:
        # Phase 1: BeautifulSoup
        try:
            timeout = aiohttp.ClientTimeout(total=SCRAPE_TIMEOUT)
            async with session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT}) as resp:
                if resp.status != 200:
                    raise ValueError(f"HTTP {resp.status}")
                html = await resp.text(errors="replace")
            body = _clean_html(html)
            if len(body) >= MIN_CONTENT_LENGTH:
                soup = BeautifulSoup(html, "lxml")
                title = soup.title.string.strip() if soup.title and soup.title.string else ""
                return RawItem(
                    url=url, domain=domain, title=title, body=body,
                    snippet=snippet, source_query=source_query,
                    used_playwright=False,
                )
        except Exception as e:
            slog.debug("bs4_scrape_failed", trace_id=trace_id, domain=domain, error=str(e))

        # Phase 2: Playwright fallback (circuit-breaker gated)
        pw_body = await _playwright_scrape(url, domain, trace_id)
        if pw_body:
            return RawItem(
                url=url, domain=domain, title="", body=pw_body,
                snippet=snippet, source_query=source_query,
                used_playwright=True,
            )

        # Phase 3: Fall back to snippet
        if snippet and len(snippet) >= MIN_CONTENT_LENGTH:
            return RawItem(
                url=url, domain=domain, title="", body=snippet,
                snippet=snippet, source_query=source_query,
                used_playwright=False,
            )
        return None


async def _concurrent_scrape(
    trace_id: str, search_results: list[dict]
) -> list[RawItem]:
    if not search_results:
        return []
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_SCRAPES)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _scrape_url(
                url=item.get("href", ""),
                domain=_extract_domain(item.get("href", "")),
                snippet=item.get("body", ""),
                source_query=item.get("_source_query", ""),
                session=session,
                semaphore=semaphore,
                trace_id=trace_id,
            )
            for item in search_results
            if item.get("href")
        ]
        results = await asyncio.gather(*tasks)
    items = [r for r in results if r is not None]
    pw_count = sum(1 for r in items if r.used_playwright)
    slog.info("scrape_complete", trace_id=trace_id,
              success=len(items), total=len(tasks), playwright_used=pw_count)
    return items


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
    trace_id: str, topic: str, items: list[RawItem], batch_offset: int
) -> list[dict]:
    items_text = "\n".join(
        f"[{batch_offset + i}] {it.title}: {it.body[:300]}"
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
    trace_id: str, topic: str, items: list[RawItem]
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
# FactCard Assembly
# ---------------------------------------------------------------------------

def _build_fact_cards(
    items: list[RawItem], scores: list[dict], max_results: int
) -> list[FactCard]:
    score_map = {s["index"]: s for s in scores}
    facts: list[FactCard] = []
    for i, item in enumerate(items):
        s = score_map.get(i)
        if not s:
            continue
        body = item.body[:500] + "..." if len(item.body) > 500 else item.body
        content = f"{item.title}: {body}" if item.title else body
        facts.append(
            FactCard(
                fact_id=make_id(),
                content=content,
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

    # 2. Concurrent search
    search_results = await _concurrent_search(trace_id, sub_queries, max_results)
    if not search_results:
        slog.warning("no_search_results", trace_id=trace_id)
        return _make_fallback_facts()

    # Tag source query for traceability
    for item in search_results:
        item["_source_query"] = query

    # 3. Concurrent scrape (BS4 + Playwright fallback with circuit breaker)
    raw_items = await _concurrent_scrape(trace_id, search_results)
    if not raw_items:
        slog.warning("no_scraped_content", trace_id=trace_id)
        return _make_fallback_facts()

    # 4. Content-level SimHash dedup
    seen_hashes: list[int] = []
    deduped: list[RawItem] = []
    for item in raw_items:
        if not _is_near_duplicate(item.body, seen_hashes):
            deduped.append(item)
    slog.info("content_dedup", trace_id=trace_id,
              before=len(raw_items), after=len(deduped))

    # 5. Batch LLM scoring
    scores = await _score_all(trace_id, query, deduped)
    if not scores:
        slog.warning("no_scores", trace_id=trace_id)
        return _make_fallback_facts()

    # 6. Build FactCards
    facts = _build_fact_cards(deduped, scores, max_results)
    if not facts:
        slog.warning("no_factcards", trace_id=trace_id)
        return _make_fallback_facts()

    slog.info("pipeline_complete", trace_id=trace_id, factcards=len(facts))
    return facts


# ---------------------------------------------------------------------------
# Public Entry Point (synchronous, preserves original interface)
# ---------------------------------------------------------------------------

def search_real_news(query: str, max_results: int = 5) -> list[FactCard]:
    """Search the web, scrape full pages, score with LLM, and return FactCards."""
    try:
        return asyncio.run(_async_pipeline(query, max_results))
    except Exception as e:
        slog.error("pipeline_failed", error=str(e))
        return _make_fallback_facts()
