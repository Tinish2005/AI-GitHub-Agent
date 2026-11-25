"""The demo seed: a ready-made index of one repo, shipped as a compressed file.

A seed file is a normal saved index (the same data the vector store keeps in
store.json) plus a small manifest. At startup the server copies it into place
when the live store is missing or empty, so a fresh server always opens on the
demo repo without paying for embeddings again.

Rules this module enforces:
    - Never overwrite or delete a live store that has data.
    - Refuse a seed that is damaged or inconsistent; the live store is left alone.
    - Write the live store through a temporary file, so a crash cannot leave it half written.
"""

from __future__ import annotations

import gzip
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from backend.indexing.vector_store import DEFAULT_COLLECTION, VectorStore


SEED_FORMAT = 1
_SMALL_STORE_BYTES = 10_000


class DemoSeedError(Exception):
    """The seed file cannot be used. The message is safe to show."""


class DemoSeedStatus(BaseModel):
    """What happened to the demo seed at startup. Shown in /stats."""

    configured: bool = Field(description="False when restoring the demo is switched off.")
    seed_found: bool = Field(default=False)
    restored: bool = Field(default=False, description="True when the demo was copied in just now.")
    message: str = Field(default="")
    warning: str = Field(default="")
    repo: str = Field(default="")
    commit: str = Field(default="")
    chunks: int = Field(default=0, ge=0)
    model: str | None = Field(default=None)
    dimension: int | None = Field(default=None)


# ---------------------------------------------------------------------------
# Building a seed
# ---------------------------------------------------------------------------


def _row(vector) -> str:
    parts = []
    for x in vector:
        f = np.float32(x)
        if not np.isfinite(f):
            raise DemoSeedError("A vector contains a value that is not a finite number.")
        parts.append(str(f))
    return "[" + ",".join(parts) + "]"


def seed_text(state: dict, manifest: dict) -> str:
    """Seed JSON text. Vectors are written at float32 precision to keep the file small."""
    vectors = state.get("vectors") or []
    embedding = state.get("embedding") or {}
    return (
        '{"format":' + str(SEED_FORMAT)
        + ',"manifest":' + json.dumps(manifest)
        + ',"ids":' + json.dumps(state["ids"])
        + ',"documents":' + json.dumps(state["documents"])
        + ',"metadatas":' + json.dumps(state["metadatas"])
        + ',"embedding":' + json.dumps(embedding)
        + ',"vectors":[' + ",".join(_row(v) for v in vectors) + "]}"
    )


def export_seed(store: VectorStore, out_path: Path, manifest_extra: dict | None = None) -> dict:
    """Write `store` as a seed file (plus a small readable manifest next to it)."""
    state_file = store._state_path()
    if not state_file.is_file():
        raise DemoSeedError("The index has nothing saved to export.")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    ids = state.get("ids") or []
    vectors = state.get("vectors") or []
    if not ids or not vectors:
        raise DemoSeedError("The index is empty, so there is nothing to export.")
    metas = state.get("metadatas") or []
    repos = sorted({m.get("repo", "") for m in metas if m.get("repo")})
    commits = sorted({m.get("commit", "") for m in metas if m.get("commit")})
    manifest = {
        "repo": repos[0] if len(repos) == 1 else ", ".join(repos),
        "commit": commits[0] if len(commits) == 1 else "",
        "chunks": len(ids),
        "files": len({m.get("file_path", "") for m in metas}),
        "dimension": len(vectors[0]),
        "model": (state.get("embedding") or {}).get("model"),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest.update(manifest_extra or {})

    text = seed_text(state, manifest)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_name(out_path.name + ".tmp")
    # No file name and mtime=0 in the header keep the bytes identical for identical content.
    with open(temp, "wb") as raw, gzip.GzipFile(
        filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=9,
    ) as gz:
        gz.write(text.encode("utf-8"))
    os.replace(temp, out_path)

    sidecar = out_path.with_name(out_path.name.replace(".json.gz", "").replace(".gz", "") + ".manifest.json")
    sidecar.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# Reading and checking a seed
# ---------------------------------------------------------------------------


def validate_seed_state(state: object) -> dict:
    """Check a parsed seed. Returns {manifest, chunks, dimension, model} or raises DemoSeedError."""
    if not isinstance(state, dict):
        raise DemoSeedError("The seed file is not a valid index.")
    if state.get("format") != SEED_FORMAT:
        raise DemoSeedError(f"Unknown seed format {state.get('format')!r} (expected {SEED_FORMAT}).")
    manifest = state.get("manifest")
    if not isinstance(manifest, dict) or not manifest.get("repo"):
        raise DemoSeedError("The seed has no manifest or no repo name.")
    lists = {k: state.get(k) for k in ("ids", "documents", "metadatas", "vectors")}
    for key, value in lists.items():
        if not isinstance(value, list):
            raise DemoSeedError(f"The seed is missing '{key}'.")
    sizes = {len(v) for v in lists.values()}
    if len(sizes) != 1:
        raise DemoSeedError("The seed lists have different lengths; the file is damaged.")
    count = sizes.pop()
    if count == 0:
        raise DemoSeedError("The seed has no chunks.")
    if len(set(state["ids"])) != count:
        raise DemoSeedError("The seed has repeated chunk ids.")
    widths = {len(v) if isinstance(v, list) else -1 for v in state["vectors"]}
    if len(widths) != 1 or widths == {0} or -1 in widths:
        raise DemoSeedError("The seed vectors are not all the same size.")
    model = (state.get("embedding") or {}).get("model") or None
    return {
        "manifest": manifest,
        "chunks": count,
        "dimension": widths.pop(),
        "model": model,
    }


def read_seed_text(path: Path) -> str:
    try:
        with gzip.open(path, "rb") as fh:
            return fh.read().decode("utf-8")
    except FileNotFoundError as exc:
        raise DemoSeedError(f"Seed file not found: {path}") from exc
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise DemoSeedError(f"The seed file cannot be read ({type(exc).__name__}).") from exc


def read_seed(path: Path) -> tuple:
    """Return (info, text) for a valid seed file, or raise DemoSeedError."""
    text = read_seed_text(path)
    try:
        state = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DemoSeedError("The seed file is not valid JSON.") from exc
    return validate_seed_state(state), text


# ---------------------------------------------------------------------------
# Restoring a seed
# ---------------------------------------------------------------------------


def live_store_state(state_file: Path) -> str:
    """'missing', 'empty', 'has_data' or 'unreadable' for the live store file."""
    if not state_file.exists():
        return "missing"
    size = state_file.stat().st_size
    if size == 0:
        return "empty"
    if size > _SMALL_STORE_BYTES:
        return "has_data"
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return "unreadable"
    return "has_data" if isinstance(data, dict) and data.get("ids") else "empty"


def _normalize_model(name: str | None) -> str | None:
    text = (name or "").strip()
    if not text:
        return None
    return text if text.startswith("models/") else f"models/{text}"


def restore_demo_seed(
    seed_path: Path,
    persist_dir: Path,
    *,
    collection: str = DEFAULT_COLLECTION,
    configured_model: str | None = None,
    enabled: bool = True,
) -> DemoSeedStatus:
    """Copy the demo seed into the live store, only when that store is missing or empty."""
    if not enabled:
        return DemoSeedStatus(configured=False, message="Restoring the demo at startup is switched off.")

    seed_path = Path(seed_path)
    if not seed_path.is_file():
        return DemoSeedStatus(
            configured=True, seed_found=False,
            message=f"No demo seed file found at {seed_path}. The demo repo will not be available.",
        )

    state_file = Path(persist_dir) / collection / "store.json"
    live = live_store_state(state_file)
    if live == "has_data":
        return DemoSeedStatus(
            configured=True, seed_found=True, restored=False,
            message="The live index already has data, so it was left unchanged.",
        )
    if live == "unreadable":
        return DemoSeedStatus(
            configured=True, seed_found=True, restored=False,
            message="The live index file exists but cannot be read. It was left unchanged.",
        )

    try:
        info, text = read_seed(seed_path)
    except DemoSeedError as exc:
        return DemoSeedStatus(
            configured=True, seed_found=True, restored=False,
            message=f"The demo seed was refused: {exc} The live index was left unchanged.",
        )

    manifest = info["manifest"]
    warning = ""
    wanted = _normalize_model(configured_model)
    if wanted and info["model"] and wanted != _normalize_model(info["model"]):
        warning = (
            f"The demo was built with model {info['model']}, but the server is set to {wanted}. "
            "Searches will fail until they match."
        )
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        temp = state_file.with_name(state_file.name + ".tmp")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, state_file)
    except OSError as exc:
        return DemoSeedStatus(
            configured=True, seed_found=True, restored=False,
            message=f"The demo could not be written ({type(exc).__name__}).",
        )
    return DemoSeedStatus(
        configured=True, seed_found=True, restored=True,
        message=f"Demo restored: {manifest['repo']} ({info['chunks']} chunks).",
        warning=warning,
        repo=str(manifest["repo"]),
        commit=str(manifest.get("commit", "")),
        chunks=info["chunks"],
        model=info["model"],
        dimension=info["dimension"],
    )
