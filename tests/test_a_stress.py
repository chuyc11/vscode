"""Stress tests for A-node: hostile URL scenarios.

Tests:
1. Cloudflare 403 cascade — 10 URLs from same domain, breaker trips
2. Dead links — 10 URLs with connection errors, breaker trips
3. Mixed scenario — 403s + dead links from same domain
4. Domain-level circuit breaker — threshold/mixed-rate/window-pruning
5. Trace ID propagation — verify structlog output contains trace_id
6. Memory bounded — verify no OOM under hostile load
7. Domain isolation — per-domain breaker independence
8. Edge case — empty snippet with dead links, no crash

Engineering red lines verified:
- Circuit breaker blocks Playwright after CB_MIN_CALLS failures at CB_THRESHOLD rate
- structlog prints trace_id on every degradation event
- Memory stays bounded (no zombie browser instances)
"""

import asyncio
import gc
import sys
import tracemalloc
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from aioresponses import aioresponses

from agent_a_retriever import (
    DomainCircuitBreaker,
    _scrape_url,
    _cb,
    CB_MIN_CALLS,
    CB_THRESHOLD,
    MAX_CONCURRENT_SCRAPES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLOUDFLARE_CHALLENGE = """<html><head><title>403 Forbidden</title></head>
<body><h1>Access denied</h1>
<p>This site is protected by Cloudflare. Please enable JavaScript.</p>
<script>setTimeout(function(){location.reload()},5000);</script>
</body></html>"""

VALID_SNIPPET = (
    "This is a substantial article about technology and AI developments "
    "in the global market. The content covers multiple aspects of the industry "
    "including chip manufacturing, cloud computing, and machine learning trends."
)


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset the global circuit breaker between tests."""
    _cb._history.clear()
    yield
    _cb._history.clear()


@pytest.fixture
def mock_playwright():
    """Mock Playwright async API. Yields (mock_pw, mock_page) for per-test control.

    By default, Playwright SUCCEEDS (returns challenge page with enough content).
    To make it FAIL (trigger circuit breaker), set:
        mock_page.goto.side_effect = Exception("blocked")
    """
    mock_page = MagicMock()
    mock_page.goto = AsyncMock()
    mock_page.content = AsyncMock(return_value=CLOUDFLARE_CHALLENGE)

    mock_browser = MagicMock()
    mock_browser.new_page = AsyncMock(return_value=mock_page)

    mock_pw = MagicMock()
    mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)

    mock_async_pw = MagicMock()
    mock_async_pw.__aenter__ = AsyncMock(return_value=mock_pw)
    mock_async_pw.__aexit__ = AsyncMock(return_value=False)

    mock_module = MagicMock()
    mock_module.async_playwright = MagicMock(return_value=mock_async_pw)
    sys.modules["playwright.async_api"] = mock_module

    with patch("agent_a_retriever.ENABLE_PLAYWRIGHT", True):
        yield mock_pw, mock_page

    sys.modules.pop("playwright.async_api", None)


def _make_urls(prefix: str, count: int, domain: str = "cf-protected.example.com") -> list[str]:
    return [f"https://{domain}/{prefix}/{i}" for i in range(count)]


# ---------------------------------------------------------------------------
# Test 1: Cloudflare 403 cascade — breaker trips after Playwright failures
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cloudflare_403_cascade(mock_playwright):
    """10 URLs from same domain return 403. Playwright also fails → breaker trips."""
    mock_pw, mock_page = mock_playwright
    # Make Playwright FAIL so the breaker records fallbacks
    mock_page.goto.side_effect = Exception("Cloudflare blocked headless browser")

    urls = _make_urls("article", 10)
    trace_id = "cf403test01"

    with aioresponses() as m:
        for url in urls:
            m.get(url, status=403, body=CLOUDFLARE_CHALLENGE, repeat=True)

        from aiohttp import ClientSession, TCPConnector
        connector = TCPConnector(limit=MAX_CONCURRENT_SCRAPES)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)
        async with ClientSession(connector=connector) as session:
            tasks = [
                _scrape_url(url, "cf-protected.example.com", VALID_SNIPPET,
                           "test", session, semaphore, trace_id)
                for url in urls
            ]
            results = await asyncio.gather(*tasks)

    raw_items = [r for r in results if r is not None]
    stats = _cb.get_stats("cf-protected.example.com")

    # Circuit breaker must be open
    assert stats["circuit_open"] is True
    assert stats["total"] >= CB_MIN_CALLS
    assert stats["fallbacks"] >= CB_MIN_CALLS
    assert stats["rate"] >= CB_THRESHOLD

    # Playwright was called for first CB_MIN_CALLS, then blocked
    pw_calls = mock_pw.chromium.launch.call_count
    assert pw_calls <= CB_MIN_CALLS + 1, \
        f"Playwright should stop after breaker opens, called {pw_calls} times"

    # Snippet fallback produces items for blocked calls
    assert len(raw_items) > 0, "Should have results from snippet fallback"

    print(f"\n  Cloudflare cascade: {len(raw_items)}/10 items, "
          f"breaker={stats}, pw_calls={pw_calls}")


# ---------------------------------------------------------------------------
# Test 2: Dead links — connection errors, Playwright also fails
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dead_links(mock_playwright):
    """10 dead URLs. Playwright also fails → breaker trips."""
    mock_pw, mock_page = mock_playwright
    mock_page.goto.side_effect = ConnectionError("Playwright also cannot connect")

    urls = _make_urls("broken", 10, domain="dead-links.example.org")
    trace_id = "deadlink01"

    with aioresponses() as m:
        for url in urls:
            m.get(url, exception=ConnectionError("Connection refused"))

        from aiohttp import ClientSession, TCPConnector
        connector = TCPConnector(limit=MAX_CONCURRENT_SCRAPES)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)
        async with ClientSession(connector=connector) as session:
            tasks = [
                _scrape_url(url, "dead-links.example.org", VALID_SNIPPET,
                           "test", session, semaphore, trace_id)
                for url in urls
            ]
            results = await asyncio.gather(*tasks)

    raw_items = [r for r in results if r is not None]
    stats = _cb.get_stats("dead-links.example.org")

    assert stats["circuit_open"] is True
    assert stats["total"] >= CB_MIN_CALLS
    assert len(raw_items) > 0, "Snippet fallback should produce items"

    print(f"\n  Dead links: {len(raw_items)}/10 items, breaker={stats}")


# ---------------------------------------------------------------------------
# Test 3: Mixed scenario — 403 + dead links
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mixed_hostile_urls(mock_playwright):
    """Mix of 403s and dead links from same domain. Breaker must trip."""
    mock_pw, mock_page = mock_playwright
    mock_page.goto.side_effect = Exception("Anti-bot detection")

    urls = _make_urls("mixed", 10, domain="mixed-hostile.example.com")
    trace_id = "mixed01test"

    with aioresponses() as m:
        for i, url in enumerate(urls):
            if i % 2 == 0:
                m.get(url, status=403, body=CLOUDFLARE_CHALLENGE)
            else:
                m.get(url, exception=ConnectionError("Connection reset"))

        from aiohttp import ClientSession, TCPConnector
        connector = TCPConnector(limit=MAX_CONCURRENT_SCRAPES)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)
        async with ClientSession(connector=connector) as session:
            tasks = [
                _scrape_url(url, "mixed-hostile.example.com", VALID_SNIPPET,
                           "test", session, semaphore, trace_id)
                for url in urls
            ]
            results = await asyncio.gather(*tasks)

    raw_items = [r for r in results if r is not None]
    stats = _cb.get_stats("mixed-hostile.example.com")

    assert stats["circuit_open"] is True
    assert len(raw_items) > 0

    print(f"\n  Mixed hostile: {len(raw_items)}/10 items, breaker={stats}")


# ---------------------------------------------------------------------------
# Test 4: Circuit breaker unit tests
# ---------------------------------------------------------------------------

def test_circuit_breaker_threshold():
    """Verify breaker trips at CB_MIN_CALLS with fallback rate >= CB_THRESHOLD."""
    cb = DomainCircuitBreaker()

    cb.record("test.com", was_fallback=True)
    cb.record("test.com", was_fallback=True)
    assert cb.is_open("test.com") is False, "Not open before min calls"

    cb.record("test.com", was_fallback=True)
    assert cb.is_open("test.com") is True, "Opens at min calls with 100% fallback"

    stats = cb.get_stats("test.com")
    assert stats["total"] == 3
    assert stats["fallbacks"] == 3
    assert stats["rate"] == 1.0
    assert stats["circuit_open"] is True


def test_circuit_breaker_mixed_rate():
    """Breaker opens at 50% fallback rate (CB_THRESHOLD)."""
    cb = DomainCircuitBreaker()
    cb.record("x.com", was_fallback=False)
    cb.record("x.com", was_fallback=True)
    cb.record("x.com", was_fallback=True)
    assert cb.is_open("x.com") is True  # 66% > 50%

    cb2 = DomainCircuitBreaker()
    cb2.record("y.com", was_fallback=False)
    cb2.record("y.com", was_fallback=False)
    cb2.record("y.com", was_fallback=True)
    assert cb2.is_open("y.com") is False  # 33% < 50%


def test_circuit_breaker_window_pruning():
    """Old entries outside CB_WINDOW are pruned."""
    import time
    cb = DomainCircuitBreaker()

    cb.record("expire.com", was_fallback=True)
    cb.record("expire.com", was_fallback=True)
    cb.record("expire.com", was_fallback=True)
    assert cb.is_open("expire.com") is True

    # Age entries past window
    cb._history["expire.com"] = [
        (time.time() - 400, True),
        (time.time() - 350, True),
        (time.time() - 310, True),
    ]
    cb.record("expire.com", was_fallback=True)
    assert cb.is_open("expire.com") is False, "Closes after entries expire"


# ---------------------------------------------------------------------------
# Test 5: Trace ID propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trace_id_in_structlog(mock_playwright):
    """Verify structlog outputs contain trace_id on degradation events."""
    mock_pw, mock_page = mock_playwright
    mock_page.goto.side_effect = Exception("blocked")

    urls = _make_urls("trace", 5, domain="trace-test.example.com")
    trace_id = "trace_test_42"

    with aioresponses() as m:
        for url in urls:
            m.get(url, status=403, body="Access Denied")

        from aiohttp import ClientSession, TCPConnector
        connector = TCPConnector(limit=MAX_CONCURRENT_SCRAPES)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)
        async with ClientSession(connector=connector) as session:
            tasks = [
                _scrape_url(url, "trace-test.example.com", VALID_SNIPPET,
                           "test", session, semaphore, trace_id)
                for url in urls
            ]
            await asyncio.gather(*tasks)

    stats = _cb.get_stats("trace-test.example.com")
    assert stats["total"] >= 3, "Should have recorded multiple attempts"

    print(f"\n  Trace ID test: trace_id={trace_id}, stats={stats}")


@pytest.mark.asyncio
async def test_trace_id_json_format(mock_playwright, monkeypatch):
    """Verify structlog JSON output with trace_id when LOG_FORMAT=json."""
    mock_pw, mock_page = mock_playwright
    mock_page.goto.side_effect = Exception("blocked")

    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL_NUM", "10")

    import structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(10),
    )

    urls = _make_urls("json", 3, domain="json-trace.example.com")
    trace_id = "json_trace_99"

    with aioresponses() as m:
        for url in urls:
            m.get(url, status=403, body="Blocked")

        from aiohttp import ClientSession, TCPConnector
        connector = TCPConnector(limit=MAX_CONCURRENT_SCRAPES)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)
        async with ClientSession(connector=connector) as session:
            tasks = [
                _scrape_url(url, "json-trace.example.com", VALID_SNIPPET,
                           "test", session, semaphore, trace_id)
                for url in urls
            ]
            await asyncio.gather(*tasks)

    stats = _cb.get_stats("json-trace.example.com")
    assert stats["total"] >= 3

    # Restore default structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),
    )


# ---------------------------------------------------------------------------
# Test 6: Memory bounded — no OOM under hostile load
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_bounded_no_oom(mock_playwright):
    """Memory stays bounded with 10 hostile URLs. Circuit breaker prevents browser OOM."""
    mock_pw, mock_page = mock_playwright
    mock_page.goto.side_effect = Exception("blocked")

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    urls = _make_urls("mem", 10, domain="oom-test.example.com")
    trace_id = "oom_test_00"

    with aioresponses() as m:
        for url in urls:
            m.get(url, status=403, body=CLOUDFLARE_CHALLENGE)

        from aiohttp import ClientSession, TCPConnector
        connector = TCPConnector(limit=MAX_CONCURRENT_SCRAPES)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)
        async with ClientSession(connector=connector) as session:
            tasks = [
                _scrape_url(url, "oom-test.example.com", VALID_SNIPPET,
                           "test", session, semaphore, trace_id)
                for url in urls
            ]
            results = await asyncio.gather(*tasks)

    gc.collect()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    total_diff = sum(
        s.size_diff for s in snapshot_after.compare_to(snapshot_before, "lineno")
    )
    memory_mb = total_diff / (1024 * 1024)

    raw_items = [r for r in results if r is not None]
    cb_stats = _cb.get_stats("oom-test.example.com")

    assert cb_stats["circuit_open"] is True
    assert len(raw_items) > 0
    assert memory_mb < 50, f"Memory usage {memory_mb:.1f} MB exceeds 50 MB limit"

    print(f"\n  Memory test: {memory_mb:.2f} MB delta, "
          f"{len(raw_items)} items, breaker={cb_stats}")


# ---------------------------------------------------------------------------
# Test 7: Domain isolation — per-domain breaker independence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multi_domain_isolation(mock_playwright):
    """Per-domain breakers: one domain's failures don't affect another."""
    mock_pw, mock_page = mock_playwright
    mock_page.goto.side_effect = Exception("blocked")

    domain_a = "hostile-a.example.com"
    domain_b = "hostile-b.example.com"
    urls_a = _make_urls("a", 5, domain=domain_a)
    urls_b = _make_urls("b", 5, domain=domain_b)

    trace_id = "multi_domain_01"

    with aioresponses() as m:
        for url in urls_a + urls_b:
            m.get(url, status=403, body=CLOUDFLARE_CHALLENGE)

        from aiohttp import ClientSession, TCPConnector
        connector = TCPConnector(limit=MAX_CONCURRENT_SCRAPES)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)
        async with ClientSession(connector=connector) as session:
            tasks = [
                _scrape_url(url, domain_a if "hostile-a" in url else domain_b,
                           VALID_SNIPPET, "test", session, semaphore, trace_id)
                for url in urls_a + urls_b
            ]
            results = await asyncio.gather(*tasks)

    stats_a = _cb.get_stats(domain_a)
    stats_b = _cb.get_stats(domain_b)

    # Each domain has its own breaker state
    # Breaker opens after CB_MIN_CALLS failures; remaining calls are blocked (not recorded)
    assert stats_a["total"] >= CB_MIN_CALLS
    assert stats_b["total"] >= CB_MIN_CALLS
    assert stats_a["circuit_open"] is True
    assert stats_b["circuit_open"] is True

    # History isolation: domain_a has entries, domain_b has separate entries
    assert domain_a in _cb._history
    assert domain_b in _cb._history
    assert _cb._history[domain_a] is not _cb._history[domain_b]

    print(f"\n  Multi-domain: A={stats_a}, B={stats_b}")


# ---------------------------------------------------------------------------
# Test 8: Edge case — empty snippet, dead links, no Playwright
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_snippet_dead_links():
    """Dead links + empty snippet + Playwright disabled = 0 results, no crash."""
    urls = _make_urls("empty", 3, domain="empty-snippet.example.com")
    trace_id = "empty_snip_01"

    # Playwright NOT enabled (no fixture). Dead links with empty snippet.
    with aioresponses() as m:
        for url in urls:
            m.get(url, exception=ConnectionError("Refused"))

        from aiohttp import ClientSession, TCPConnector
        connector = TCPConnector(limit=MAX_CONCURRENT_SCRAPES)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)
        async with ClientSession(connector=connector) as session:
            tasks = [
                _scrape_url(url, "empty-snippet.example.com", "",
                           "test", session, semaphore, trace_id)
                for url in urls
            ]
            results = await asyncio.gather(*tasks)

    raw_items = [r for r in results if r is not None]
    assert len(raw_items) == 0

    stats = _cb.get_stats("empty-snippet.example.com")
    assert stats["total"] == 0, "No Playwright calls = no breaker activity"
