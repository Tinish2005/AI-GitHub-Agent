"""Unit tests for `backend.indexing.embeddings`."""

from __future__ import annotations

import pytest

from backend.indexing.embeddings import (
    DEFAULT_MODEL_NAME,
    EMBEDDING_DIMENSION,
    EmbeddingService,
)


class _FakeBackend:
    """Deterministic fake encoder for fast unit tests."""

    def __init__(self, dim: int = EMBEDDING_DIMENSION) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[list[float]]:
        self.calls.append(list(sentences))
        # Embedding = [len(s), len(s)+1, ...] truncated to `dim`.
        out: list[list[float]] = []
        for s in sentences:
            base = float(len(s))
            out.append([base + i for i in range(self.dim)])
        return out


def test_embed_text_returns_list_of_floats() -> None:
    fake = _FakeBackend()
    svc = EmbeddingService(backend=fake)
    vec = svc.embed_text("hello")
    assert isinstance(vec, list)
    assert all(isinstance(x, float) for x in vec)
    assert len(vec) == EMBEDDING_DIMENSION


def test_embed_texts_returns_one_vector_per_input() -> None:
    fake = _FakeBackend()
    svc = EmbeddingService(backend=fake)
    vecs = svc.embed_texts(["a", "bb", "ccc"])
    assert len(vecs) == 3
    assert all(len(v) == EMBEDDING_DIMENSION for v in vecs)


def test_embed_texts_empty_input_returns_empty_list() -> None:
    svc = EmbeddingService(backend=_FakeBackend())
    assert svc.embed_texts([]) == []


def test_embed_text_rejects_empty_string() -> None:
    svc = EmbeddingService(backend=_FakeBackend())
    with pytest.raises(ValueError):
        svc.embed_text("")


def test_embed_texts_rejects_any_empty_string() -> None:
    svc = EmbeddingService(backend=_FakeBackend())
    with pytest.raises(ValueError):
        svc.embed_texts(["valid", ""])


def test_default_model_name_constant_is_minilm() -> None:
    assert DEFAULT_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"


def test_default_dimension_is_384() -> None:
    assert EMBEDDING_DIMENSION == 384


def test_backend_is_called_once_per_batch() -> None:
    fake = _FakeBackend()
    svc = EmbeddingService(backend=fake)
    svc.embed_texts(["one", "two", "three"])
    assert len(fake.calls) == 1
    assert fake.calls[0] == ["one", "two", "three"]