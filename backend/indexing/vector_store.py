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
import logging
import os
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.indexing.embeddings import EmbeddingService
from backend.indexing.models import ChunkMetadata, CodeChunk


DEFAULT_COLLECTION: str = "code_chunks"
_STATE_FILENAME: str = "store.json"

logger = logging.getLogger(__name__)


class EmbeddingMismatchError(RuntimeError):
    """The embedding model in use does not match the one the index was built with.

    Nothing is changed or deleted when this is raised. The message is safe to show.
    """


# ----------------------------------------------------------------------
# Shared in-memory copy of store.json
# ----------------------------------------------------------------------
# The API builds a new VectorStore for every request, and parsing the index file
# is the slow part. So the parsed file is kept here, once per process, and every
# VectorStore picks it up. Before each use the file is checked (one os.stat call);
# when it has changed (an indexing job finished, the demo was restored, anything
# else wrote it), it is read again. The cached copy is never changed in place.

_SNAPSHOT_LOCK = threading.Lock()
_SNAPSHOTS: dict[str, "_Snapshot"] = {}


@dataclass(frozen=True)
class _Snapshot:
    """One parsed read of store.json. Read-only: nobody may change it in place."""

    signature: tuple
    ids: tuple
    documents: tuple
    metadatas: tuple
    vectors: np.ndarray | None  # marked read-only
    embedding_model: str | None


def _file_signature(path: Path) -> tuple | None:
    """Cheap fingerprint of the file on disk, or None when it does not exist.

    mtime alone can miss a rewrite on a coarse clock, so size and file id are included.
    The indexer replaces the file with os.replace, which also changes the file id.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _read_state_file(path: Path) -> dict:
    """Read and parse store.json. (Tests count calls to this to prove the cache works.)"""
    return json.loads(path.read_text(encoding="utf-8"))


def _build_snapshot(data: dict, signature: tuple) -> _Snapshot:
    vectors = data.get("vectors", [])
    array = np.asarray(vectors, dtype=np.float32) if vectors else None
    if array is not None:
        array.setflags(write=False)
    embedding = data.get("embedding") or {}
    return _Snapshot(
        signature=signature,
        ids=tuple(data.get("ids", [])),
        documents=tuple(data.get("documents", [])),
        metadatas=tuple(dict(m) for m in data.get("metadatas", [])),
        vectors=array,
        embedding_model=embedding.get("model") or None,
    )


def _load_snapshot(path: Path, signature: tuple) -> _Snapshot:
    """The parsed file, from memory when it has not changed since it was read.

    If the file cannot be parsed (for example it was caught half-written) and an
    older good copy exists, that copy keeps being served and the read is retried
    on the next call. With no older copy the error is raised, as before.
    """
    key = str(path)
    with _SNAPSHOT_LOCK:
        cached = _SNAPSHOTS.get(key)
        if cached is not None and cached.signature == signature:
            return cached
        try:
            snapshot = _build_snapshot(_read_state_file(path), signature)
        except (OSError, ValueError) as exc:
            if cached is None:
                raise
            logger.warning(
                "The index file could not be read (%s). Serving the previous copy.",
                type(exc).__name__,
            )
            return cached
        _SNAPSHOTS[key] = snapshot
        return snapshot


def clear_index_cache() -> None:
    """Forget every cached index. Used by tests."""
    with _SNAPSHOT_LOCK:
        _SNAPSHOTS.clear()


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
        self._embedding_model: str | None = None  # model the vectors came from
        self._loaded: bool = False
        self._signature: tuple | None = None  # file fingerprint this state matches

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[CodeChunk],
        metadatas: list[ChunkMetadata] | None = None,
        *,
        extra_metadata: dict[str, str] | None = None,
    ) -> list[str]:
        """
        Upsert chunks into the collection. Returns the IDs used.

        `extra_metadata` (str -> str) is saved with every chunk in this call.
        The indexers use it to record the repo, branch and index time.

        If `metadatas` is provided it must have the same length as
        `chunks`; otherwise lightweight metadata is derived from each
        chunk on the fly.

        A chunk's id is built from the repo label, file, function name and its
        position among chunks with the same name in that file. So identical
        code in two places stays separate, and indexing again updates a chunk
        in place. If the embedding model or vector size differs from what the
        index already holds, `EmbeddingMismatchError` is raised and nothing is
        changed.
        """
        if not chunks:
            return []
        if metadatas is not None and len(metadatas) != len(chunks):
            raise ValueError(
                f"metadatas length ({len(metadatas)}) must match chunks length ({len(chunks)})"
            )

        self._ensure_loaded()
        self._own_vectors()

        documents = [c.source for c in chunks]
        repo_label = str((extra_metadata or {}).get("repo", ""))
        ids = _chunk_ids(chunks, repo_label)
        meta_dicts = [
            _meta_to_dict(chunks[i], metadatas[i] if metadatas else None)
            for i in range(len(chunks))
        ]
        if extra_metadata:
            tags = {str(k): str(v) for k, v in extra_metadata.items()}
            for meta in meta_dicts:
                meta.update(tags)
        new_vectors = np.asarray(
            self._embeddings.embed_texts(documents), dtype=np.float32
        )
        new_vectors = _l2_normalize(new_vectors)

        # Never mix models and never delete the existing index: refuse instead.
        current_model = self._embeddings.active_model_name
        self._check_compatible(int(new_vectors.shape[1]), current_model)
        was_empty = not self._ids

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

        if was_empty:
            # Remember the model only for a fresh index, so we never claim a
            # model for vectors we did not create.
            self._embedding_model = current_model
        self._persist()
        return ids

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        repo: str | None = None,
    ) -> list[QueryResult]:
        """Return the top-k nearest chunks to `text`.

        With `repo` ("owner/name"), only chunks labelled with that repo are
        searched. Chunks without a label never match a repo-scoped search.
        """
        if not text:
            raise ValueError("Query text must be non-empty.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        self._ensure_loaded()
        if not self._ids or self._vectors is None or self._vectors.size == 0:
            return []

        q = np.asarray(self._embeddings.embed_text(text), dtype=np.float32)
        q = _l2_normalize(q.reshape(1, -1))[0]

        # A model or size mismatch used to return an empty list silently. Say so.
        self._check_compatible(int(q.shape[0]), self._embeddings.active_model_name)

        # Cosine similarity = dot product when both sides are L2-normalized.
        similarities = self._vectors @ q
        # Convert to a distance in [0, 2]; smaller = closer.
        distances = 1.0 - similarities

        candidates = len(self._ids)
        if repo is not None and repo.strip():
            wanted = repo.strip().lower()
            mask = np.array(
                [str(m.get("repo", "")).strip().lower() == wanted for m in self._metadatas],
                dtype=bool,
            )
            candidates = int(mask.sum())
            if candidates == 0:
                return []
            distances = np.where(mask, distances, np.inf)

        k = min(top_k, candidates)
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

    def all_metadata(self) -> list[dict[str, str]]:
        """Return a copy of every chunk's metadata (read-only, no embeddings)."""
        self._ensure_loaded()
        return [dict(m) for m in self._metadatas]

    def repo_summaries(self) -> dict:
        """{"owner/name": {"chunks": n, "demo": bool, "indexed_at": str}} for labelled chunks."""
        self._ensure_loaded()
        summaries: dict = {}
        for meta in self._metadatas:
            label = str(meta.get("repo", "")).strip()
            if not label:
                continue
            entry = summaries.setdefault(label, {"chunks": 0, "demo": False, "indexed_at": ""})
            entry["chunks"] += 1
            if str(meta.get("demo", "")).lower() == "true":
                entry["demo"] = True
            entry["indexed_at"] = max(entry["indexed_at"], str(meta.get("indexed_at", "")))
        return summaries

    def remove_repo(self, label: str) -> int:
        """Delete every chunk labelled with this repo. Returns how many were removed."""
        self._ensure_loaded()
        wanted = label.strip().lower()
        keep = [
            i for i, meta in enumerate(self._metadatas)
            if str(meta.get("repo", "")).strip().lower() != wanted
        ]
        removed = len(self._ids) - len(keep)
        if removed == 0:
            return 0
        self._ids = [self._ids[i] for i in keep]
        self._documents = [self._documents[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]
        self._vectors = self._vectors[keep] if (self._vectors is not None and keep) else None
        self._persist()
        return removed

    def embedding_info(self) -> dict:
        """The model and vector size this index was built with (None if unknown)."""
        self._ensure_loaded()
        dimension = None
        if self._vectors is not None and self._vectors.size:
            dimension = int(self._vectors.shape[1])
        return {"model": self._embedding_model, "dimension": dimension}

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
        self._embedding_model = None
        self._loaded = True
        state_file = self._state_path()
        if state_file.exists():
            state_file.unlink()
        with _SNAPSHOT_LOCK:
            _SNAPSHOTS.pop(str(state_file), None)
        self._signature = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _state_path(self) -> Path:
        return self.persist_directory / self.collection_name / _STATE_FILENAME

    def _ensure_loaded(self) -> None:
        """Make this store match the file on disk, reading it only when it changed."""
        state_file = self._state_path()
        signature = _file_signature(state_file)
        if self._loaded and signature == self._signature:
            return
        if signature is None:
            self._ids, self._documents, self._metadatas = [], [], []
            self._vectors = None
            self._embedding_model = None
        else:
            snapshot = _load_snapshot(state_file, signature)
            self._ids = list(snapshot.ids)
            self._documents = list(snapshot.documents)
            self._metadatas = list(snapshot.metadatas)  # dicts are shared, never edited in place
            self._vectors = snapshot.vectors  # shared and read-only; writers copy it first
            self._embedding_model = snapshot.embedding_model
            signature = snapshot.signature
        self._signature = signature
        self._loaded = True

    def _own_vectors(self) -> None:
        """Before changing vectors in place, take a private copy of the shared read-only array."""
        if self._vectors is not None and not self._vectors.flags.writeable:
            self._vectors = self._vectors.copy()

    def _check_compatible(self, dimension: int, model: str | None) -> None:
        """Raise EmbeddingMismatchError if the vectors cannot be mixed with the index."""
        if not self._ids or self._vectors is None or not self._vectors.size:
            return
        stored_dim = int(self._vectors.shape[1])
        stored_model = self._embedding_model
        wrong_size = stored_dim != dimension
        wrong_model = bool(stored_model and model and stored_model != model)
        if not (wrong_size or wrong_model):
            return
        raise EmbeddingMismatchError(
            "The search index does not match the embedding model in use. "
            f"The index has model {stored_model or 'unknown'} with {stored_dim} numbers per chunk; "
            f"the current model is {model or 'unknown'} with {dimension}. "
            "Nothing was changed. Use the same model (set GEMINI_EMBEDDING_MODEL), "
            "or rebuild the index on purpose."
        )
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
        if self._embedding_model:
            payload["embedding"] = {"model": self._embedding_model}
        # Write to a temp file, then swap it in, so a crash cannot leave a half-written store.
        text = json.dumps(payload)
        temp_file = state_file.with_name(state_file.name + ".tmp")
        temp_file.write_text(text, encoding="utf-8")
        os.replace(temp_file, state_file)
        self._signature = _file_signature(state_file)  # our own write is not an outside change


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _hash(source: str) -> str:
    """Local fallback content hash, matching `metadata._content_hash`."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _chunk_ids(chunks: list[CodeChunk], repo_label: str) -> list[str]:
    """Stable, unique ids: repo + file + function name + position among same names."""
    seen: Counter = Counter()
    repo = repo_label.strip().lower()
    ids: list[str] = []
    for chunk in chunks:
        path = str(chunk.file_path).replace("\\", "/")
        key = (path, chunk.qualified_name)
        position = seen[key]
        seen[key] += 1
        raw = f"{repo}\n{path}\n{chunk.qualified_name}\n{position}"
        ids.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())
    return ids


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
