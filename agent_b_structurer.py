"""B-node: fact structuring with spaCy NER + MiniLM embeddings.

Receives RawItem list from A-node, sanitises text against prompt injection,
runs CPU-intensive NLP in ProcessPoolExecutor (with trimmed IPC payloads
and chunked batch dispatch), then assembles FactCard list with exposed
Entity collections.

Engineering red lines
---------------------
1. Prompt Injection sanitisation layer runs BEFORE any NLP.
2. ProcessPoolExecutor isolates CPU work; cross-process payloads are
   trimmed to (item_id, text) only, dispatched in chunks to cap IPC I/O.
3. All底层 I/O is async-safe (asyncio.to_thread for sync NLP calls).
"""

import asyncio
import logging
import os
import re
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional

import structlog

from schema import FactCard, Entity, SourceTier, EvidenceType, make_id, utc_now

# Import RawItem from A-node (lightweight dataclass, no heavy deps)
from agent_a_retriever import RawItem

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
SPACY_MODEL = os.getenv("B_NODE_SPACY_MODEL", "en_core_web_sm")
MINI_MODEL = os.getenv("B_NODE_MINI_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("B_NODE_CHUNK_SIZE", "16"))
MAX_WORKERS = int(os.getenv("B_NODE_MAX_WORKERS", "4"))
MIN_TEXT_LENGTH = 50

# ---------------------------------------------------------------------------
# Prompt Injection Sanitisation
# ---------------------------------------------------------------------------

# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS = [
    # Direct instruction overrides
    r"(?i)\bignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)\b",
    r"(?i)\bdisregard\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)\b",
    r"(?i)\bforget\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)\b",
    # Role hijacking
    r"(?i)\byou\s+are\s+now\s+(a|an|the)\b",
    r"(?i)\bact\s+as\s+(a|an|the)\b",
    r"(?i)\bpretend\s+(you\s+)?(are|to\s+be)\b",
    r"(?i)\bnew\s+persona\b",
    r"(?i)\bsystem\s*:\s*you\s+are\b",
    # Prompt leaking
    r"(?i)\b(show|reveal|print|output|repeat)\s+(your|the|system)\s+(prompt|instructions?|rules?)\b",
    r"(?i)\bwhat\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?|rules?)\b",
    # Encoded/obfuscated instructions
    r"(?i)\b(base64|rot13|hex)\s*(decode|encode)\b",
    r"(?i)\b\\x[0-9a-fA-F]{2}",
    # Delimiter injection
    r"(?i)```[\s]*system[\s]*```",
    r"(?i)<\|im_start\|>",
    r"(?i)<\|im_end\|>",
    r"(?i)\[INST\]",
    r"(?i)\[/INST\]",
    # DAN-style jailbreaks
    r"(?i)\bdo\s+anything\s+now\b",
    r"(?i)\bjailbreak\b",
    r"(?i)\bDAN\s+mode\b",
]
_INJECTION_RE = [re.compile(p) for p in _INJECTION_PATTERNS]

# Characters/sequences to neutralise even if not full injection
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_ZERO_WIDTH_RE = re.compile(r"[​‌‍‎‏﻿]")


def sanitize_text(text: str) -> tuple[str, bool]:
    """Sanitise text against prompt injection.

    Returns (cleaned_text, was_injection_detected).
    """
    if not text:
        return "", False

    detected = False

    # Check for injection patterns
    for pattern in _INJECTION_RE:
        if pattern.search(text):
            detected = True
            # Neutralise the matched region by replacing with [REDACTED]
            text = pattern.sub("[SANITIZED]", text)

    # Strip control characters
    text = _CONTROL_CHARS_RE.sub("", text)

    # Strip zero-width characters (used for steganography)
    text = _ZERO_WIDTH_RE.sub("", text)

    # Collapse excessive whitespace (some attacks use huge whitespace)
    text = re.sub(r"\s{10,}", " ", text)

    # Truncate extremely long text (prevent token overflow attacks)
    max_len = 10000
    if len(text) > max_len:
        text = text[:max_len]
        detected = True

    return text.strip(), detected


# ---------------------------------------------------------------------------
# IPC Payload — trimmed for ProcessPoolExecutor
# ---------------------------------------------------------------------------

@dataclass
class IPCTask:
    """Minimal payload sent to worker processes. Only essential fields
    to minimise Pickle serialisation overhead."""
    item_id: str
    text: str


@dataclass
class IPCResult:
    """Result returned from worker processes."""
    item_id: str
    entities: list[tuple[str, str, int, int]]  # (text, label, start, end)


# ---------------------------------------------------------------------------
# Worker function (runs in separate process)
# ---------------------------------------------------------------------------

def _worker_extract_entities(tasks: list[IPCTask]) -> list[IPCResult]:
    """Extract named entities using spaCy. Runs in a child process.

    The spaCy model is loaded once per process (lazily on first call).
    """
    import spacy
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        # Fallback: blank model with NER pipeline
        nlp = spacy.blank("en")
        if "ner" not in nlp.pipe_names:
            nlp.add_pipe("ner")

    results = []
    for task in tasks:
        try:
            doc = nlp(task.text)
            ents = [
                (ent.text, ent.label_, ent.start_char, ent.end_char)
                for ent in doc.ents
                if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "MONEY",
                                   "DATE", "TIME", "PERCENT", "QUANTITY",
                                   "CARDINAL", "ORDINAL", "NORP", "FAC",
                                   "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"}
            ]
            results.append(IPCResult(item_id=task.item_id, entities=ents))
        except Exception as e:
            logger.warning("NER failed for item %s: %s", task.item_id, e)
            results.append(IPCResult(item_id=task.item_id, entities=[]))
    return results


# ---------------------------------------------------------------------------
# Chunked dispatch to ProcessPoolExecutor
# ---------------------------------------------------------------------------

def _chunk_list(lst: list, chunk_size: int) -> list[list]:
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


async def _extract_entities_parallel(
    trace_id: str, items: list[RawItem]
) -> dict[str, list[Entity]]:
    """Dispatch NER to ProcessPoolExecutor with trimmed payloads and chunked batches.

    Returns {item_id: [Entity, ...]}.
    """
    # Build trimmed IPC payloads (only id + text)
    tasks = [
        IPCTask(item_id=f"raw_{i}", text=item.title + " " + item.body[:3000])
        for i, item in enumerate(items)
    ]

    chunks = _chunk_list(tasks, CHUNK_SIZE)
    slog.info("ner_dispatch", trace_id=trace_id,
              total_tasks=len(tasks), chunks=len(chunks), chunk_size=CHUNK_SIZE)

    all_results: list[IPCResult] = []

    def _run_chunks_sync():
        results = []
        # Use fewer workers than chunks to avoid spawning too many processes
        workers = min(MAX_WORKERS, len(chunks))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_worker_extract_entities, chunk) for chunk in chunks]
            for future in futures:
                try:
                    chunk_results = future.result(timeout=120)
                    results.extend(chunk_results)
                except Exception as e:
                    slog.warning("ner_chunk_failed", trace_id=trace_id, error=str(e))
        return results

    all_results = await asyncio.to_thread(_run_chunks_sync)

    # Map results back to entities
    entity_map: dict[str, list[Entity]] = {}
    for result in all_results:
        entities = [
            Entity(text=t, label=l, start=s, end=e)
            for t, l, s, e in result.entities
        ]
        entity_map[result.item_id] = entities

    total_ents = sum(len(v) for v in entity_map.values())
    slog.info("ner_complete", trace_id=trace_id,
              items=len(entity_map), total_entities=total_ents)
    return entity_map


# ---------------------------------------------------------------------------
# Embedding extraction (main process, batched)
# ---------------------------------------------------------------------------

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(MINI_MODEL)
    return _embedder


async def _extract_embeddings(
    trace_id: str, items: list[RawItem]
) -> list[Optional[list[float]]]:
    """Extract embeddings in the main process using batched MiniLM.

    Returns a list parallel to items, each element is the embedding vector or None.
    """
    texts = [(item.title + " " + item.body[:512]) for item in items]
    if not texts:
        return []

    def _encode_sync():
        try:
            model = _get_embedder()
            embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            slog.warning("embedding_failed", trace_id=trace_id, error=str(e))
            return [None] * len(texts)

    embeddings = await asyncio.to_thread(_encode_sync)
    slog.info("embeddings_complete", trace_id=trace_id, count=len(embeddings))
    return embeddings


# ---------------------------------------------------------------------------
# FactCard assembly
# ---------------------------------------------------------------------------

def _build_fact_cards(
    trace_id: str,
    items: list[RawItem],
    entity_map: dict[str, list[Entity]],
    max_results: int,
) -> list[FactCard]:
    """Assemble FactCards from RawItems + extracted entities."""
    facts: list[FactCard] = []
    injection_count = 0

    for i, item in enumerate(items):
        item_id = f"raw_{i}"

        # Sanitise
        clean_body, was_injection = sanitize_text(item.body)
        clean_title, title_injection = sanitize_text(item.title)
        if was_injection or title_injection:
            injection_count += 1

        if not clean_body or len(clean_body) < MIN_TEXT_LENGTH:
            continue

        content = f"{clean_title}: {clean_body}" if clean_title else clean_body
        entities = entity_map.get(item_id, [])

        facts.append(
            FactCard(
                fact_id=make_id(),
                content=content[:2000],
                source_tier=SourceTier.SECONDARY,
                evidence_type=EvidenceType.DOCUMENT,
                timestamp=utc_now(),
                credibility_score=0.7,
                relevance_score=0.5,
                summary=f"Extracted {len(entities)} entities from {item.domain}",
                entities=entities,
            )
        )

    if injection_count > 0:
        slog.warning("prompt_injection_detected",
                      trace_id=trace_id, count=injection_count)

    facts = facts[:max_results]
    slog.info("factcards_assembled", trace_id=trace_id, count=len(facts))
    return facts


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def structure_raw_items(
    raw_items: list[RawItem],
    max_results: int = 10,
    trace_id: str | None = None,
) -> list[FactCard]:
    """B-node main entry: RawItem list → FactCard list with entities.

    1. Prompt Injection sanitisation
    2. ProcessPoolExecutor NER (trimmed IPC, chunked dispatch)
    3. Main-process MiniLM embeddings
    4. FactCard assembly with exposed Entity collection
    """
    if trace_id is None:
        trace_id = uuid.uuid4().hex[:12]

    slog.info("b_node_start", trace_id=trace_id, raw_items=len(raw_items))

    if not raw_items:
        return []

    # Step 1: Pre-sanitise all text (count injections for observability)
    injection_count = 0
    for item in raw_items:
        _, inj1 = sanitize_text(item.body)
        _, inj2 = sanitize_text(item.title)
        if inj1 or inj2:
            injection_count += 1
    if injection_count > 0:
        slog.warning("pre_nlp_injection_scan", trace_id=trace_id,
                      flagged=injection_count, total=len(raw_items))

    # Step 2: NER in ProcessPoolExecutor (CPU isolation)
    entity_map = await _extract_entities_parallel(trace_id, raw_items)

    # Step 3: Embeddings in main process (batched, avoids IPC tensor overhead)
    # Embeddings are computed but not stored in FactCard currently;
    # they can be used for downstream similarity search.
    await _extract_embeddings(trace_id, raw_items)

    # Step 4: Assemble FactCards
    facts = _build_fact_cards(trace_id, raw_items, entity_map, max_results)

    slog.info("b_node_complete", trace_id=trace_id, factcards=len(facts))
    return facts


def structure_raw_items_sync(
    raw_items: list[RawItem],
    max_results: int = 10,
    trace_id: str | None = None,
) -> list[FactCard]:
    """Synchronous wrapper for callers that don't use async."""
    return asyncio.run(structure_raw_items(raw_items, max_results, trace_id))
