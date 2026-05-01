"""C-node: financial enrichment with OpenBB Platform.

Receives FactCard dict (with Entity collections from B-node), maps ORG
entities to stock tickers with context-aware validation, fetches
quantitative data via OpenBB SDK, and outputs QuantFactCard list.

Engineering red lines
---------------------
1. Context-Aware Validation: LLM maps entity→ticker AND validates that
   the entity's context in the FactCard summary aligns with the ticker.
   e.g. "Apple" in a fruit context must NOT map to AAPL.
   Below confidence threshold → treated as no-data (skip).
2. Synchronous OpenBB SDK calls run in asyncio.to_thread (thread pool).
3. tenacity retry with wait_random_exponential on rate-limit / transient
   failures. Complete failure → whitebox structlog, skip (never crash).
4. All I/O is async-safe; structlog with Trace ID throughout.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Optional

import structlog
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)

from schema import (
    FactCard, Entity, QuantFactCard, TickerMapping,
    PriceData, FundamentalData, make_id, utc_now,
)
from llm_client import call_llm_json

# ---------------------------------------------------------------------------
# Structlog
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if os.getenv("LOG_FORMAT") != "json"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        int(os.getenv("LOG_LEVEL_NUM", "20"))
    ),
)
slog = structlog.get_logger()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAPPING_CONFIDENCE_THRESHOLD = float(
    os.getenv("C_NODE_CONFIDENCE_THRESHOLD", "0.6")
)
MAX_CONCURRENT_FETCHES = int(os.getenv("C_NODE_MAX_CONCURRENT", "3"))
OPENBB_PROVIDER = os.getenv("C_NODE_OPENBB_PROVIDER", "yfinance")

# ---------------------------------------------------------------------------
# Context-Aware Ticker Mapping via LLM
# ---------------------------------------------------------------------------

_MAPPING_SYSTEM_PROMPT = """You are a financial entity-to-ticker mapper.

Given an entity name and its surrounding context from a news article, determine:
1. The correct stock ticker symbol (if the entity is a publicly traded company)
2. The canonical company name
3. A confidence score (0.0-1.0) indicating how certain you are
4. Whether the entity context truly aligns with the financial ticker

CRITICAL: If the entity name is ambiguous (e.g. "Apple" could be fruit or tech),
you MUST check the context. If the context is about fruit/agriculture, do NOT
map to AAPL. If context is about technology/consumer electronics, map to AAPL.

If the entity is NOT a publicly traded company, return ticker=null.

Respond in JSON:
{
  "ticker": "AAPL" or null,
  "company_name": "Apple Inc.",
  "confidence": 0.95,
  "context_aligned": true,
  "reason": "Entity refers to Apple Inc. in context of iPhone launch, not fruit"
}"""

_MAPPING_USER_TEMPLATE = """Entity: {entity_text}
Entity label: {entity_label}
Context (from article): {context}

Map this entity to a stock ticker if applicable."""


async def _map_entity_to_ticker(
    trace_id: str,
    entity: Entity,
    fact_content: str,
    fact_summary: str,
) -> Optional[TickerMapping]:
    """Map a single ORG entity to a stock ticker with context validation.

    Returns TickerMapping if confident, None if entity is not mappable
    or context doesn't align.
    """
    # Build context: summary + excerpt around entity
    context = fact_summary if fact_summary else fact_content[:500]

    messages = [
        {"role": "system", "content": _MAPPING_SYSTEM_PROMPT},
        {"role": "user", "content": _MAPPING_USER_TEMPLATE.format(
            entity_text=entity.text,
            entity_label=entity.label,
            context=context,
        )},
    ]

    schema = {
        "name": "ticker_mapping",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": ["string", "null"]},
                "company_name": {"type": "string"},
                "confidence": {"type": "number"},
                "context_aligned": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["ticker", "company_name", "confidence", "context_aligned", "reason"],
            "additionalProperties": False,
        },
    }

    try:
        raw = call_llm_json(messages, schema)
        data = json.loads(raw)

        if not data.get("ticker"):
            slog.debug("entity_not_ticker", trace_id=trace_id,
                       entity=entity.text, label=entity.label)
            return None

        confidence = float(data.get("confidence", 0))
        context_aligned = data.get("context_aligned", False)

        if confidence < MAPPING_CONFIDENCE_THRESHOLD or not context_aligned:
            slog.info("entity_mapping_rejected", trace_id=trace_id,
                      entity=entity.text, ticker=data.get("ticker"),
                      confidence=confidence, context_aligned=context_aligned,
                      reason=data.get("reason", ""))
            return None

        return TickerMapping(
            entity_text=entity.text,
            ticker=data["ticker"].upper(),
            company_name=data.get("company_name", ""),
            validation_confidence=confidence,
            validation_reason=data.get("reason", ""),
            context_aligned=context_aligned,
        )
    except Exception as e:
        slog.warning("ticker_mapping_failed", trace_id=trace_id,
                     entity=entity.text, error=str(e))
        return None


# ---------------------------------------------------------------------------
# OpenBB Data Fetching (with tenacity retry)
# ---------------------------------------------------------------------------

def _is_openbb_retryable(exc: BaseException) -> bool:
    """Retry on connection/timeout/rate-limit errors."""
    exc_str = str(exc).lower()
    if "rate" in exc_str and "limit" in exc_str:
        return True
    if "too many requests" in exc_str:
        return True
    if "timeout" in exc_str or "timed out" in exc_str:
        return True
    if "connection" in exc_str:
        return True
    return False


@retry(
    retry=retry_if_exception(_is_openbb_retryable),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch_price_sync(ticker: str, provider: str) -> PriceData:
    """Fetch price data via OpenBB. Runs in thread pool."""
    from openbb import obb

    result = obb.equity.price.quote(symbol=ticker, provider=provider)
    if not result.results:
        return PriceData()

    q = result.results[0]
    return PriceData(
        last_price=getattr(q, "last_price", None),
        prev_close=getattr(q, "prev_close", None),
        open_price=getattr(q, "open", None),
        day_high=getattr(q, "high", None),
        day_low=getattr(q, "low", None),
        volume=getattr(q, "volume", None),
        market_cap=getattr(q, "market_cap", None),
        fifty_two_week_high=getattr(q, "fifty_two_week_high", None),
        fifty_two_week_low=getattr(q, "fifty_two_week_low", None),
        currency=getattr(q, "currency", "USD") or "USD",
    )


@retry(
    retry=retry_if_exception(_is_openbb_retryable),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch_fundamentals_sync(ticker: str, provider: str) -> FundamentalData:
    """Fetch fundamental metrics via OpenBB. Runs in thread pool."""
    from openbb import obb

    try:
        result = obb.equity.fundamental.ratios(symbol=ticker, provider=provider)
        if not result.results:
            return FundamentalData()

        r = result.results[0]
        return FundamentalData(
            pe_ratio=getattr(r, "price_earnings_ratio", None),
            pb_ratio=getattr(r, "price_to_book_ratio", None),
            ps_ratio=getattr(r, "price_to_sales_ratio", None),
            dividend_yield=getattr(r, "dividend_yield", None),
            revenue_ttm=getattr(r, "revenue_per_share_ttm", None),
            net_income_ttm=getattr(r, "net_income_per_share_ttm", None),
            eps_ttm=getattr(r, "earnings_per_share_ttm", None),
            roe=getattr(r, "return_on_equity", None),
            debt_to_equity=getattr(r, "debt_to_equity", None),
            current_ratio=getattr(r, "current_ratio", None),
            gross_margin=getattr(r, "gross_profit_margin", None),
            operating_margin=getattr(r, "operating_profit_margin", None),
            beta=getattr(r, "beta", None),
        )
    except Exception:
        # Ratios endpoint may not be available for all providers
        return FundamentalData()


async def _fetch_financial_data(
    trace_id: str,
    ticker: str,
    provider: str,
) -> tuple[PriceData, FundamentalData, list[str]]:
    """Fetch price + fundamentals concurrently in thread pool.

    Returns (price, fundamentals, errors).
    """
    errors: list[str] = []

    async def _price():
        try:
            return await asyncio.to_thread(_fetch_price_sync, ticker, provider)
        except Exception as e:
            errors.append(f"price: {e}")
            slog.warning("openbb_price_failed", trace_id=trace_id,
                         ticker=ticker, error=str(e))
            return PriceData()

    async def _fundamentals():
        try:
            return await asyncio.to_thread(
                _fetch_fundamentals_sync, ticker, provider
            )
        except Exception as e:
            errors.append(f"fundamentals: {e}")
            slog.warning("openbb_fundamentals_failed", trace_id=trace_id,
                         ticker=ticker, error=str(e))
            return FundamentalData()

    price, fundamentals = await asyncio.gather(_price(), _fundamentals())
    return price, fundamentals, errors


# ---------------------------------------------------------------------------
# Deduplication: same ticker from multiple FactCards → fetch once
# ---------------------------------------------------------------------------

def _deduplicate_mappings(
    mappings: list[tuple[str, TickerMapping]],
) -> dict[str, tuple[str, TickerMapping]]:
    """Deduplicate by ticker, keeping highest confidence mapping.

    Returns {ticker: (fact_id, mapping)}.
    """
    best: dict[str, tuple[str, TickerMapping]] = {}
    for fact_id, mapping in mappings:
        existing = best.get(mapping.ticker)
        if existing is None or mapping.validation_confidence > existing[1].validation_confidence:
            best[mapping.ticker] = (fact_id, mapping)
    return best


# ---------------------------------------------------------------------------
# QuantFactCard assembly
# ---------------------------------------------------------------------------

def _build_quant_cards(
    trace_id: str,
    ticker_data: dict[str, tuple[str, TickerMapping, PriceData, FundamentalData, list[str]]],
) -> list[QuantFactCard]:
    """Assemble QuantFactCards from fetched data."""
    cards: list[QuantFactCard] = []
    for ticker, (fact_id, mapping, price, fundamentals, errors) in ticker_data.items():
        cards.append(
            QuantFactCard(
                source_fact_id=fact_id,
                ticker=ticker,
                company_name=mapping.company_name,
                mapping_confidence=mapping.validation_confidence,
                mapping_reason=mapping.validation_reason,
                price=price,
                fundamentals=fundamentals,
                fetch_success=len(errors) == 0,
                fetch_errors=errors,
            )
        )
    return cards


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def enrich_financial_data(
    facts: dict[str, FactCard],
    trace_id: str | None = None,
) -> list[QuantFactCard]:
    """C-node main entry: FactCard dict → QuantFactCard list.

    1. Extract ORG entities from all FactCards
    2. LLM-based entity→ticker mapping with context-aware validation
    3. Deduplicate tickers (keep highest confidence)
    4. Concurrent OpenBB fetch in thread pool (price + fundamentals)
    5. Assemble QuantFactCards
    """
    if trace_id is None:
        trace_id = uuid.uuid4().hex[:12]

    slog.info("c_node_start", trace_id=trace_id, facts=len(facts))

    if not facts:
        return []

    # Step 1: Collect all ORG entities across all facts
    org_entities: list[tuple[str, Entity]] = []
    for fact_id, fact in facts.items():
        for entity in fact.entities:
            if entity.label == "ORG":
                org_entities.append((fact_id, entity))

    if not org_entities:
        slog.info("c_node_no_org_entities", trace_id=trace_id)
        return []

    slog.info("c_node_org_entities", trace_id=trace_id, count=len(org_entities))

    # Step 2: Map entities to tickers (concurrent LLM calls)
    mapping_tasks = []
    for fact_id, entity in org_entities:
        fact = facts[fact_id]
        mapping_tasks.append(
            _map_entity_to_ticker(trace_id, entity, fact.content, fact.summary)
        )

    mapping_results = await asyncio.gather(*mapping_tasks, return_exceptions=True)

    # Collect valid mappings
    raw_mappings: list[tuple[str, TickerMapping]] = []
    for (fact_id, entity), result in zip(org_entities, mapping_results):
        if isinstance(result, Exception):
            slog.warning("mapping_exception", trace_id=trace_id,
                         entity=entity.text, error=str(result))
            continue
        if result is not None:
            raw_mappings.append((fact_id, result))

    if not raw_mappings:
        slog.info("c_node_no_valid_mappings", trace_id=trace_id)
        return []

    slog.info("c_node_mappings", trace_id=trace_id,
              total=len(raw_mappings),
              tickers=len(set(m.ticker for _, m in raw_mappings)))

    # Step 3: Deduplicate tickers
    unique_tickers = _deduplicate_mappings(raw_mappings)
    slog.info("c_node_dedup", trace_id=trace_id,
              unique_tickers=len(unique_tickers))

    # Step 4: Fetch financial data concurrently (semaphore-bounded)
    sem = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    async def _fetch_with_sem(ticker: str, fact_id: str, mapping: TickerMapping):
        async with sem:
            price, fundamentals, errors = await _fetch_financial_data(
                trace_id, ticker, OPENBB_PROVIDER
            )
            return ticker, fact_id, mapping, price, fundamentals, errors

    fetch_tasks = [
        _fetch_with_sem(ticker, fact_id, mapping)
        for ticker, (fact_id, mapping) in unique_tickers.items()
    ]
    fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    # Build ticker_data dict
    ticker_data: dict[str, tuple[str, TickerMapping, PriceData, FundamentalData, list[str]]] = {}
    for result in fetch_results:
        if isinstance(result, Exception):
            slog.warning("fetch_exception", trace_id=trace_id, error=str(result))
            continue
        ticker, fact_id, mapping, price, fundamentals, errors = result
        ticker_data[ticker] = (fact_id, mapping, price, fundamentals, errors)

    # Step 5: Assemble QuantFactCards
    quant_cards = _build_quant_cards(trace_id, ticker_data)

    success_count = sum(1 for c in quant_cards if c.fetch_success)
    slog.info("c_node_complete", trace_id=trace_id,
              quant_cards=len(quant_cards), fetch_success=success_count)

    return quant_cards


def enrich_financial_data_sync(
    facts: dict[str, FactCard],
    trace_id: str | None = None,
) -> list[QuantFactCard]:
    """Synchronous wrapper for callers that don't use async."""
    return asyncio.run(enrich_financial_data(facts, trace_id))
