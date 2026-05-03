"""Unit tests for F-node (agent_f_reader.py).

Tests cover:
- FactCard assembly with SourceMetadata provenance
- Source tier and credibility score for internal documents
- Absolute UTC timestamps
- Empty directory handling
- IPC task/chunk dataclass construction
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import FactCard, SourceMetadata, SourceTier, EvidenceType
from agent_f_reader import (
    FIPCTask,
    FIPCChunk,
    _build_fact_cards,
    read_local_documents,
    read_local_documents_sync,
)


# ---------------------------------------------------------------------------
# IPC dataclass tests
# ---------------------------------------------------------------------------

class TestIPCTask:
    def test_create_task(self):
        task = FIPCTask(
            doc_id="doc_0",
            text="Hello world",
            file_name="test.txt",
            page_number=None,
            file_path="/tmp/test.txt",
        )
        assert task.doc_id == "doc_0"
        assert task.text == "Hello world"
        assert task.file_name == "test.txt"
        assert task.page_number is None

    def test_task_with_page_number(self):
        task = FIPCTask(
            doc_id="doc_1",
            text="Page content",
            file_name="report.pdf",
            page_number=5,
            file_path="/docs/report.pdf",
        )
        assert task.page_number == 5


class TestIPCChunk:
    def test_create_chunk(self):
        chunk = FIPCChunk(
            doc_id="doc_0",
            chunk_text="some text",
            chunk_index=0,
            file_name="test.txt",
            page_number=None,
            embedding=[0.1, 0.2, 0.3],
        )
        assert chunk.chunk_index == 0
        assert len(chunk.embedding) == 3

    def test_chunk_without_embedding(self):
        chunk = FIPCChunk(
            doc_id="doc_0",
            chunk_text="some text",
            chunk_index=0,
            file_name="test.txt",
            page_number=3,
        )
        assert chunk.embedding is None
        assert chunk.page_number == 3


# ---------------------------------------------------------------------------
# FactCard assembly tests
# ---------------------------------------------------------------------------

class TestBuildFactCards:
    def test_basic_assembly(self):
        chunks = [
            FIPCChunk(
                doc_id="doc_0",
                chunk_text="NVIDIA 预计 Q1 营收 240 亿美元",
                chunk_index=0,
                file_name="sample_report.txt",
                page_number=None,
                embedding=[0.1] * 384,
            ),
        ]
        facts = _build_fact_cards("test_trace", chunks, max_chunks=50)

        assert len(facts) == 1
        fact = facts[0]
        assert isinstance(fact, FactCard)
        assert fact.source_tier == SourceTier.PRIMARY
        assert fact.evidence_type == EvidenceType.DOCUMENT
        assert fact.credibility_score == 0.85
        assert "NVIDIA" in fact.content

    def test_source_metadata_provenance(self):
        chunks = [
            FIPCChunk(
                doc_id="doc_0",
                chunk_text="Test content",
                chunk_index=2,
                file_name="report.pdf",
                page_number=7,
                embedding=[0.1] * 384,
            ),
        ]
        facts = _build_fact_cards("test_trace", chunks, max_chunks=50)

        assert len(facts) == 1
        meta = facts[0].source_metadata
        assert meta is not None
        assert isinstance(meta, SourceMetadata)
        assert meta.file_name == "report.pdf"
        assert meta.page_number == 7
        assert meta.chunk_index == 2

    def test_absolute_timestamp(self):
        before = datetime.now(timezone.utc)
        chunks = [
            FIPCChunk(
                doc_id="doc_0",
                chunk_text="Timestamp test",
                chunk_index=0,
                file_name="test.txt",
                page_number=None,
            ),
        ]
        facts = _build_fact_cards("test_trace", chunks, max_chunks=50)
        after = datetime.now(timezone.utc)

        assert len(facts) == 1
        assert before <= facts[0].timestamp <= after

    def test_summary_contains_provenance(self):
        chunks = [
            FIPCChunk(
                doc_id="doc_0",
                chunk_text="Content",
                chunk_index=3,
                file_name="market_brief.md",
                page_number=None,
            ),
        ]
        facts = _build_fact_cards("test_trace", chunks, max_chunks=50)

        assert "[内参]" in facts[0].summary
        assert "market_brief.md" in facts[0].summary
        assert "chunk 3" in facts[0].summary

    def test_max_chunks_limit(self):
        chunks = [
            FIPCChunk(
                doc_id=f"doc_{i}",
                chunk_text=f"Chunk {i}",
                chunk_index=i,
                file_name="test.txt",
                page_number=None,
            )
            for i in range(100)
        ]
        facts = _build_fact_cards("test_trace", chunks, max_chunks=5)
        assert len(facts) == 5

    def test_content_truncation(self):
        long_text = "A" * 3000
        chunks = [
            FIPCChunk(
                doc_id="doc_0",
                chunk_text=long_text,
                chunk_index=0,
                file_name="test.txt",
                page_number=None,
            ),
        ]
        facts = _build_fact_cards("test_trace", chunks, max_chunks=50)
        assert len(facts[0].content) <= 2000

    def test_empty_chunks(self):
        facts = _build_fact_cards("test_trace", [], max_chunks=50)
        assert facts == []


# ---------------------------------------------------------------------------
# Integration tests (require LlamaIndex installed)
# ---------------------------------------------------------------------------

class TestReadLocalDocuments:
    @pytest.fixture
    def test_docs_dir(self):
        return os.path.join(os.path.dirname(__file__), "test_docs")

    def test_sync_wrapper(self, test_docs_dir):
        """Test synchronous entry point loads real documents."""
        facts = read_local_documents_sync(test_docs_dir, max_chunks=20)
        assert len(facts) > 0
        for fact in facts:
            assert isinstance(fact, FactCard)
            assert fact.source_tier == SourceTier.PRIMARY
            assert fact.credibility_score == 0.85
            assert fact.source_metadata is not None
            assert fact.source_metadata.file_name != ""

    def test_file_names_present(self, test_docs_dir):
        """Verify file_name metadata is populated."""
        facts = read_local_documents_sync(test_docs_dir, max_chunks=20)
        file_names = {f.source_metadata.file_name for f in facts}
        assert any("sample_report" in fn for fn in file_names)

    def test_empty_directory(self, tmp_path):
        """Empty directory returns empty list."""
        facts = read_local_documents_sync(str(tmp_path), max_chunks=50)
        assert facts == []

    def test_nonexistent_directory(self):
        """Non-existent directory returns empty list."""
        facts = read_local_documents_sync("/nonexistent/path/abc123", max_chunks=50)
        assert facts == []
