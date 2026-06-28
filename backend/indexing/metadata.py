"""
Metadata extractor for `CodeChunk` objects.

For each chunk we compute:
    - A SHA-256 content hash (for change detection + dedup in later loops).
    - The dotted module path derived from the file path.
    - The set of top-level imports visible in the file the chunk lives in.
    - The set of function/method call names referenced inside the chunk.

Design notes:
    - File-level imports are computed once per file and cached, so a file
      with N chunks is only re-parsed once.
    - All returned collections are tuples (immutable, hashable) sorted
      alphabetically for deterministic output.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from backend.indexing.models import ChunkMetadata, CodeChunk


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_metadata(
    chunk: CodeChunk,
    *,
    project_root: Path | None = None,
) -> ChunkMetadata:
    """
    Build a `ChunkMetadata` for a single chunk.

    `project_root`, when provided, is used to compute the chunk's dotted
    `module_path` (e.g. `backend.indexing.ast_parser`). When omitted, the
    module path falls back to the file's stem (e.g. `ast_parser`).
    """
    file_imports = _extract_file_imports(chunk.file_path)
    chunk_calls = _extract_calls(chunk.source)
    return ChunkMetadata(
        qualified_name=chunk.qualified_name,
        file_path=chunk.file_path,
        module_path=_derive_module_path(chunk.file_path, project_root),
        content_hash=_content_hash(chunk.source),
        imports=file_imports,
        calls=chunk_calls,
    )


def extract_metadata_for_chunks(
    chunks: list[CodeChunk],
    *,
    project_root: Path | None = None,
) -> list[ChunkMetadata]:
    """
    Build metadata for a batch of chunks, caching file-level imports.

    Two chunks from the same file share the same `imports` tuple, so we
    parse each file's imports exactly once.
    """
    imports_cache: dict[Path, tuple[str, ...]] = {}
    result: list[ChunkMetadata] = []

    for chunk in chunks:
        file_imports = imports_cache.get(chunk.file_path)
        if file_imports is None:
            file_imports = _extract_file_imports(chunk.file_path)
            imports_cache[chunk.file_path] = file_imports

        result.append(
            ChunkMetadata(
                qualified_name=chunk.qualified_name,
                file_path=chunk.file_path,
                module_path=_derive_module_path(chunk.file_path, project_root),
                content_hash=_content_hash(chunk.source),
                imports=file_imports,
                calls=_extract_calls(chunk.source),
            )
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _content_hash(source: str) -> str:
    """Return the lowercase hex SHA-256 of the chunk source."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _derive_module_path(file_path: Path, project_root: Path | None) -> str:
    """
    Convert a file path into a dotted Python module path.

    Example:
        file_path    = /repo/backend/indexing/ast_parser.py
        project_root = /repo
        result       = "backend.indexing.ast_parser"

    Falls back to the file's stem when project_root is missing or unrelated.
    """
    if project_root is not None:
        try:
            relative = file_path.resolve().relative_to(project_root.resolve())
        except ValueError:
            return file_path.stem
        parts = list(relative.with_suffix("").parts)
        # Strip a trailing __init__ — modules are addressed by their package.
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts) if parts else file_path.stem
    return file_path.stem


def _extract_file_imports(file_path: Path) -> tuple[str, ...]:
    """
    Return the deduped, sorted tuple of top-level imports for a file.

    Handles both `import x` and `from a.b import c` styles. Missing or
    unparseable files yield an empty tuple instead of raising.
    """
    if not file_path.is_file():
        return ()

    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return ()

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)

    return tuple(sorted(names))


def _extract_calls(source: str) -> tuple[str, ...]:
    """
    Return the deduped, sorted tuple of call names found in a chunk.

    For `foo()` we record "foo"; for `obj.bar()` we record "bar".
    Unparseable source yields an empty tuple.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)

    return tuple(sorted(names))