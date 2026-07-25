"""
Embedding model wrapper.

Wraps `sentence-transformers/all-MiniLM-L6-v2` (384-dim) into a small,
testable interface. The model is loaded lazily and cached so we don't
re-load it for every embedding call.

Design notes:
    - Lazy + cached loading: importing this module is cheap; the heavy
      model only loads on first `embed_*` call.
    - Batched encoding by default: avoids one-by-one calls for large
      chunk lists.
    - Returns plain Python `list[float]` (not NumPy arrays) so the
      values can be JSON-serialized and stored in ChromaDB without
      extra conversion.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol


DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384


class SupportsEncode(Protocol):
    """Minimal interface any embedding backend must satisfy."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = ...,
        show_progress_bar: bool = ...,
        convert_to_numpy: bool = ...,
    ) -> object:
        ...


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SupportsEncode:
    """
    Load and cache a sentence-transformers model.

    Import is local so importing this module does not pull in the
    (heavy) sentence-transformers dependency until needed.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingService:
    """
    High-level facade for turning text into vectors.

    Use one instance for the lifetime of the app; the underlying model
    is cached at module level so multiple instances are still cheap.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        batch_size: int = 32,
        backend: SupportsEncode | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        # Allow injecting a fake backend for tests.
        self._backend: SupportsEncode | None = backend

    @property
    def backend(self) -> SupportsEncode:
        """Lazily resolve the backend (real model or injected fake)."""
        if self._backend is None:
            self._backend = _load_model(self.model_name)
        return self._backend

    def embed_text(self, text: str) -> list[float]:
        """Encode a single string and return its embedding as a plain list."""
        if not text:
            raise ValueError("Cannot embed an empty string.")
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of strings and return embeddings as plain lists."""
        if not texts:
            return []
        if any(not t for t in texts):
            raise ValueError("All texts must be non-empty.")

        raw = self.backend.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Convert NumPy / tensor output into pure Python lists.
        return [[float(x) for x in vector] for vector in raw]