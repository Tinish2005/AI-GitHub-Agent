"""Integration tests for `backend.rag.pipeline`."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.indexing.ast_parser import parse_source
from backend.indexing.embeddings import EMBEDDING_DIMENSION, EmbeddingService
from backend.indexing.vector_store import VectorStore
from backend.rag.llm import EchoLLMClient
from backend.rag.models import Answer
from backend.rag.pipeline import RAGPipeline


SAMPLE_SOURCE = '''\
"""Sample module."""

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def greet(name: str) -> str:
    """Greet someone politely."""
    return f"Hello, {name}"


class Calculator:
    """A tiny calculator."""

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b
'''


class _FakeBackend:
    """Deterministic length-based encoder for fast tests."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[list[float]]:
        out: list[list[float]] = []
        for s in sentences:
            base = float(len(s))
            out.append([base + i for i in range(EMBEDDING_DIMENSION)])
        return out


@pytest.fixture
def populated_store(tmp_path: Path) -> VectorStore:
    svc = EmbeddingService(backend=_FakeBackend())
    store = VectorStore(
        persist_directory=tmp_path / "vdb",
        collection_name=f"rag_{tmp_path.name}",
        embedding_service=svc,
    )
    chunks = parse_source(SAMPLE_SOURCE, Path("sample.py"))
    store.add_chunks(chunks)
    return store


def test_ask_returns_answer(populated_store: VectorStore) -> None:
    pipeline = RAGPipeline(vector_store=populated_store, llm=EchoLLMClient())
    out = pipeline.ask("How do I add two numbers?")
    assert isinstance(out, Answer)
    assert out.question == "How do I add two numbers?"
    assert out.answer  # non-empty
    assert out.model == "echo-test"


def test_ask_includes_sources(populated_store: VectorStore) -> None:
    pipeline = RAGPipeline(vector_store=populated_store, llm=EchoLLMClient(), top_k=3)
    out = pipeline.ask("greet someone")
    assert out.source_count >= 1
    assert all(s.chunk_id for s in out.sources)
    assert all(s.qualified_name for s in out.sources)


def test_ask_rejects_empty_question(populated_store: VectorStore) -> None:
    pipeline = RAGPipeline(vector_store=populated_store, llm=EchoLLMClient())
    with pytest.raises(ValueError):
        pipeline.ask("   ")


def test_pipeline_rejects_zero_top_k(populated_store: VectorStore) -> None:
    with pytest.raises(ValueError):
        RAGPipeline(vector_store=populated_store, llm=EchoLLMClient(), top_k=0)


def test_used_context_chars_is_positive_when_sources_exist(populated_store: VectorStore) -> None:
    pipeline = RAGPipeline(vector_store=populated_store, llm=EchoLLMClient())
    out = pipeline.ask("multiply")
    assert out.used_context_chars > 0


def test_ask_with_empty_store_still_answers(tmp_path: Path) -> None:
    """Even with no indexed chunks, the pipeline should not crash."""
    svc = EmbeddingService(backend=_FakeBackend())
    empty_store = VectorStore(
        persist_directory=tmp_path / "empty",
        collection_name=f"empty_{tmp_path.name}",
        embedding_service=svc,
    )
    pipeline = RAGPipeline(vector_store=empty_store, llm=EchoLLMClient())
    out = pipeline.ask("anything")
    assert out.source_count == 0
    assert out.used_context_chars == 0