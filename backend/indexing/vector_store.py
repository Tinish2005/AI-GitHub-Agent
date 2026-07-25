"""
Lightweight, pure-Python vector store.

A simple but production-shaped replacement for ChromaDB that works
flawlessly on Python 3.14:

    - Vectors and metadata are kept in memory for fast queries.
    - State is persisted to a single JSON file on disk.
    - Cosine similarity is computed with NumPy.
    - The public API mirrors what we'd use from any real vector DB,
      so swapping back to Chroma / Qdrant / LanceDB later is a
      drop-in change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.indexing.embeddings import EmbeddingService
from backend.indexing.models import ChunkMetadata, CodeChunk


DEFAULT_COLLECTION: str = "code_chunks"
_STATE_FILENAME: str = "store.json"


@dataclass(frozen=True)
class QueryResult:
    """A single hit from a similarity query."""

    chunk_id: str
    document: str
    metadata: dict[str, str]
    distance: float


class VectorStore:
    """
    File-backed vector store with a Chroma-compatible surface.

    Construction does **not** touch disk until the first call that
    actually needs to read or write state.
    """

    def __init__(
        self,
        persist_directory: Path,
        *,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._embeddings = embedding_service or EmbeddingService()
        # State (lazy-loaded)
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict[str, str]] = []
        self._vectors: np.ndarray | None = None  # shape (N, D)
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[CodeChunk],
        metadatas: list[ChunkMetadata] | None = None,
    ) -> list[str]:
        """
        Upsert chunks into the collection. Returns the IDs used.

        If `metadatas` is provided it must have the same length as
        `chunks`; otherwise lightweight metadata is derived from each
        chunk on the fly.
        """
        if not chunks:
            return []
        if metadatas is not None and len(metadatas) != len(chunks):
            raise ValueError(
                f"metadatas length ({len(metadatas)}) must match chunks length ({len(chunks)})"
            )

        self._ensure_loaded()

        documents = [c.source for c in chunks]
        ids = [
            metadatas[i].content_hash if metadatas else _hash(c.source)
            for i, c in enumerate(chunks)
        ]
        meta_dicts = [
            _meta_to_dict(chunks[i], metadatas[i] if metadatas else None)
            for i in range(len(chunks))
        ]
        new_vectors = np.asarray(
            self._embeddings.embed_texts(documents), dtype=np.float32
        )
        new_vectors = _l2_normalize(new_vectors)

        # Upsert: replace existing entries by id, append the rest.
        id_to_index = {cid: i for i, cid in enumerate(self._ids)}
        for i, cid in enumerate(ids):
            if cid in id_to_index:
                idx = id_to_index[cid]
                self._documents[idx] = documents[i]
                self._metadatas[idx] = meta_dicts[i]
                assert self._vectors is not None
                self._vectors[idx] = new_vectors[i]
            else:
                self._ids.append(cid)
                self._documents.append(documents[i])
                self._metadatas.append(meta_dicts[i])
                if self._vectors is None:
                    self._vectors = new_vectors[i : i + 1].copy()
                else:
                    self._vectors = np.vstack([self._vectors, new_vectors[i : i + 1]])

        self._persist()
        return ids

    def query(self, text: str, *, top_k: int = 5) -> list[QueryResult]:
        """Return the top-k nearest chunks to `text`."""
        if not text:
            raise ValueError("Query text must be non-empty.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        self._ensure_loaded()
        if not self._ids or self._vectors is None or self._vectors.size == 0:
            return []

        q = np.asarray(self._embeddings.embed_text(text), dtype=np.float32)
        q = _l2_normalize(q.reshape(1, -1))[0]

        # Cosine similarity = dot product when both sides are L2-normalized.
        similarities = self._vectors @ q
        # Convert to a distance in [0, 2]; smaller = closer.
        distances = 1.0 - similarities

        k = min(top_k, len(self._ids))
        # argpartition is O(N); take top-k smallest distances and then sort them.
        partitioned = np.argpartition(distances, k - 1)[:k]
        order = partitioned[np.argsort(distances[partitioned])]

        return [
            QueryResult(
                chunk_id=self._ids[i],
                document=self._documents[i],
                metadata=dict(self._metadatas[i]),
                distance=float(distances[i]),
            )
            for i in order
        ]

    def count(self) -> int:
        """Number of items currently stored."""
        self._ensure_loaded()
        return len(self._ids)

    def reset(self) -> None:
        """Delete the entire collection (irreversible)."""
        self._ids = []
        self._documents = []
        self._metadatas = []
        self._vectors = None
        self._loaded = True
        state_file = self._state_path()
        if state_file.exists():
            state_file.unlink()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _state_path(self) -> Path:
        return self.persist_directory / self.collection_name / _STATE_FILENAME

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        state_file = self._state_path()
        if state_file.is_file():
            data = json.loads(state_file.read_text(encoding="utf-8"))
            self._ids = list(data.get("ids", []))
            self._documents = list(data.get("documents", []))
            self._metadatas = [dict(m) for m in data.get("metadatas", [])]
            vectors = data.get("vectors", [])
            self._vectors = (
                np.asarray(vectors, dtype=np.float32) if vectors else None
            )
        self._loaded = True

    def _persist(self) -> None:
        state_file = self._state_path()
        state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ids": self._ids,
            "documents": self._documents,
            "metadatas": self._metadatas,
            "vectors": (
                self._vectors.tolist() if self._vectors is not None else []
            ),
        }
        state_file.write_text(json.dumps(payload), encoding="utf-8")


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _hash(source: str) -> str:
    """Local fallback content hash, matching `metadata._content_hash`."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _meta_to_dict(
    chunk: CodeChunk,
    metadata: ChunkMetadata | None,
) -> dict[str, str]:
    """
    Flatten chunk + (optional) metadata into a simple str→str dict that
    is JSON-serializable. Tuples are joined with ',' for round-tripping.
    """
    out: dict[str, str] = {
        "qualified_name": chunk.qualified_name,
        "name": chunk.name,
        "chunk_type": chunk.chunk_type.value,
        "file_path": str(chunk.file_path),
        "start_line": str(chunk.start_line),
        "end_line": str(chunk.end_line),
        "language": chunk.language,
    }
    if chunk.parent is not None:
        out["parent"] = chunk.parent
    if metadata is not None:
        out["module_path"] = metadata.module_path
        out["content_hash"] = metadata.content_hash 
        if metadata.imports:
            out["imports"] = ",".join(metadata.imports)
        if metadata.calls:
            out["calls"] = ",".join(metadata.calls)
    return out


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. Zero rows are returned as zeros."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms