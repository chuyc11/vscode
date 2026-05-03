"""F-node: local internal document reader via LlamaIndex.

Loads local documents (PDF/TXT/MD), chunks and vectorises them inside
ProcessPoolExecutor (following the B-node IPC pattern), then assembles
FactCards with full knowledge provenance (file_name, page_number) and
absolute UTC timestamps.

Engineering red lines
---------------------
1. ProcessPoolExecutor isolates CPU-bound chunking + embedding work.
2. Cross-process payloads are trimmed to (doc_id, text, file_name,
   page_number, file_path) only, dispatched in chunks to cap IPC I/O.
3. Every FactCard carries SourceMetadata for knowledge provenance.
4. All FactCard timestamps use utc_now() (absolute, not relative).
"""

import asyncio
import logging
import os
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import structlog

from schema import (
    FactCard,
    SourceMetadata,
    SourceTier,
    EvidenceType,
    make_id,
    utc_now,
)

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
CHUNK_SIZE = int(os.getenv("F_NODE_CHUNK_SIZE", "16"))
MAX_WORKERS = int(os.getenv("F_NODE_MAX_WORKERS", "4"))
EMBED_MODEL_NAME = os.getenv("F_NODE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLAMA_CHUNK_SIZE = int(os.getenv("F_NODE_LLAMA_CHUNK_SIZE", "512"))
LLAMA_CHUNK_OVERLAP = int(os.getenv("F_NODE_LLAMA_CHUNK_OVERLAP", "50"))
MAX_CHUNKS = int(os.getenv("F_NODE_MAX_CHUNKS", "50"))
CREDIBILITY_SCORE = 0.85  # 内参默认可信度高于网络来源

# ---------------------------------------------------------------------------
# IPC Payloads — trimmed for ProcessPoolExecutor
# ---------------------------------------------------------------------------

@dataclass
class FIPCTask:
    """Minimal payload sent to worker processes."""
    doc_id: str
    text: str
    file_name: str
    page_number: Optional[int]
    file_path: str


@dataclass
class FIPCChunk:
    """Result returned from worker processes."""
    doc_id: str
    chunk_text: str
    chunk_index: int
    file_name: str
    page_number: Optional[int]
    embedding: Optional[list] = None


# ---------------------------------------------------------------------------
# Document loading (main process)
# ---------------------------------------------------------------------------

def _load_documents(doc_dir: str) -> list:
    """Load local documents using LlamaIndex SimpleDirectoryReader.

    Supports PDF, TXT, MD. PDF pages are extracted with page metadata.
    """
    from llama_index.core.readers import SimpleDirectoryReader

    if not os.path.isdir(doc_dir):
        raise ValueError(f"Document directory does not exist: {doc_dir}")

    reader = SimpleDirectoryReader(
        input_dir=doc_dir,
        recursive=True,
        required_exts=[".pdf", ".txt", ".md"],
    )
    documents = reader.load_data()
    slog.info("documents_loaded", dir=doc_dir, count=len(documents))
    return documents


# ---------------------------------------------------------------------------
# Worker function (runs in separate process)
# ---------------------------------------------------------------------------

def _worker_chunk_and_embed(tasks: list[FIPCTask]) -> list[FIPCChunk]:
    """Chunk documents and compute embeddings in a child process.

    LlamaIndex SentenceSplitter handles chunking; sentence_transformers
    handles embedding (from local cache, no HuggingFace hub calls needed).
    """
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import Document
    from sentence_transformers import SentenceTransformer

    splitter = SentenceSplitter(
        chunk_size=LLAMA_CHUNK_SIZE,
        chunk_overlap=LLAMA_CHUNK_OVERLAP,
    )
    # Use sentence_transformers directly — loads from local cache,
    # avoids HuggingFaceEmbedding's hub connectivity check in subprocess.
    st_model = SentenceTransformer(EMBED_MODEL_NAME.split("/")[-1])

    results: list[FIPCChunk] = []
    for task in tasks:
        try:
            doc = Document(
                text=task.text,
                metadata={
                    "file_name": task.file_name,
                    "page_number": task.page_number,
                    "file_path": task.file_path,
                },
            )
            nodes = splitter.get_nodes_from_documents([doc])

            # Batch-embed all chunks using sentence_transformers
            texts = [n.get_content() for n in nodes]
            raw_embeddings = st_model.encode(texts, batch_size=32, show_progress_bar=False)
            embeddings = [emb.tolist() for emb in raw_embeddings]

            for i, (node, emb) in enumerate(zip(nodes, embeddings)):
                # Extract page number from node metadata (PDF may attach page_label)
                page = (
                    node.metadata.get("page_label")
                    or node.metadata.get("page_number")
                    or task.page_number
                )
                page_int = int(page) if page is not None else None

                results.append(FIPCChunk(
                    doc_id=task.doc_id,
                    chunk_text=node.get_content(),
                    chunk_index=i,
                    file_name=str(node.metadata.get("file_name", task.file_name)),
                    page_number=page_int,
                    embedding=emb,
                ))
        except Exception as e:
            logger.warning("chunk_embed_failed for doc %s: %s", task.doc_id, e)

    return results


# ---------------------------------------------------------------------------
# Chunked dispatch to ProcessPoolExecutor
# ---------------------------------------------------------------------------

def _chunk_list(lst: list, chunk_size: int) -> list[list]:
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


async def _chunk_and_embed_parallel(
    trace_id: str, tasks: list[FIPCTask]
) -> list[FIPCChunk]:
    """Dispatch chunking + embedding to ProcessPoolExecutor.

    Uses trimmed IPC payloads and chunked batch dispatch, mirroring B-node.
    """
    chunks = _chunk_list(tasks, CHUNK_SIZE)
    slog.info("f_node_dispatch", trace_id=trace_id,
              total_tasks=len(tasks), chunks=len(chunks), chunk_size=CHUNK_SIZE)

    def _run_chunks_sync():
        results: list[FIPCChunk] = []
        workers = min(MAX_WORKERS, len(chunks))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_worker_chunk_and_embed, chunk) for chunk in chunks]
            for future in futures:
                try:
                    chunk_results = future.result(timeout=300)
                    results.extend(chunk_results)
                except Exception as e:
                    slog.warning("f_node_chunk_failed", trace_id=trace_id, error=str(e))
        return results

    all_chunks = await asyncio.to_thread(_run_chunks_sync)
    slog.info("f_node_complete", trace_id=trace_id,
              total_chunks=len(all_chunks), docs=len(set(c.doc_id for c in all_chunks)))
    return all_chunks


# ---------------------------------------------------------------------------
# FactCard assembly
# ---------------------------------------------------------------------------

def _build_fact_cards(
    trace_id: str,
    chunks: list[FIPCChunk],
    max_chunks: int,
) -> list[FactCard]:
    """Assemble FactCards from IPC chunks with full knowledge provenance."""
    facts: list[FactCard] = []

    for chunk in chunks[:max_chunks]:
        # Build summary with provenance
        page_str = f"p.{chunk.page_number}" if chunk.page_number is not None else "p.-"
        summary = f"[内参] {chunk.file_name} {page_str} — chunk {chunk.chunk_index}"

        # Build source metadata for knowledge provenance
        source_meta = SourceMetadata(
            file_name=chunk.file_name,
            file_path="",
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
        )

        facts.append(
            FactCard(
                fact_id=make_id(),
                content=chunk.chunk_text[:2000],
                source_tier=SourceTier.PRIMARY,
                evidence_type=EvidenceType.DOCUMENT,
                timestamp=utc_now(),
                credibility_score=CREDIBILITY_SCORE,
                relevance_score=0.7,
                summary=summary,
                entities=[],
                source_metadata=source_meta,
            )
        )

    slog.info("f_node_factcards_assembled", trace_id=trace_id, count=len(facts))
    return facts


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def read_local_documents(
    doc_dir: str,
    query: str = "",
    max_chunks: int = MAX_CHUNKS,
    trace_id: Optional[str] = None,
) -> list[FactCard]:
    """F-node main entry: local document directory -> FactCard list.

    Pipeline:
    1. SimpleDirectoryReader loads PDF/TXT/MD documents
    2. Build trimmed IPC payloads (FIPCTask)
    3. ProcessPoolExecutor chunking + embedding (chunked dispatch)
    4. Assemble FactCards with SourceMetadata provenance and utc_now() timestamps
    """
    if trace_id is None:
        trace_id = uuid.uuid4().hex[:12]

    slog.info("f_node_start", trace_id=trace_id, doc_dir=doc_dir)

    # Step 1: Load documents in main process
    try:
        documents = _load_documents(doc_dir)
    except Exception as e:
        slog.error("f_node_load_failed", trace_id=trace_id, error=str(e))
        return []

    if not documents:
        slog.warning("f_node_no_documents", trace_id=trace_id)
        return []

    # Step 2: Build IPC tasks from loaded documents
    tasks: list[FIPCTask] = []
    for i, doc in enumerate(documents):
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        tasks.append(FIPCTask(
            doc_id=f"doc_{i}",
            text=doc.text if hasattr(doc, "text") else str(doc),
            file_name=str(meta.get("file_name", f"document_{i}")),
            page_number=meta.get("page_label") or meta.get("page_number"),
            file_path=str(meta.get("file_path", "")),
        ))

    slog.info("f_node_tasks_built", trace_id=trace_id, task_count=len(tasks))

    # Step 3: Chunk + embed in ProcessPoolExecutor
    ipc_chunks = await _chunk_and_embed_parallel(trace_id, tasks)

    if not ipc_chunks:
        slog.warning("f_node_no_chunks", trace_id=trace_id)
        return []

    # Step 4: Assemble FactCards
    facts = _build_fact_cards(trace_id, ipc_chunks, max_chunks)

    slog.info("f_node_done", trace_id=trace_id, factcards=len(facts))
    return facts


def read_local_documents_sync(
    doc_dir: str,
    query: str = "",
    max_chunks: int = MAX_CHUNKS,
    trace_id: Optional[str] = None,
) -> list[FactCard]:
    """Synchronous wrapper for callers that don't use async."""
    return asyncio.run(read_local_documents(doc_dir, query, max_chunks, trace_id))
