"""
AST-based parser for Python source files.

Walks a Python file's abstract syntax tree and produces a list of `CodeChunk`s
— one per top-level function, top-level class, and per method inside any class.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

from backend.indexing.models import ChunkType, CodeChunk

# Type alias for "anything we treat as a function-like node".
FuncNode = ast.FunctionDef | ast.AsyncFunctionDef


def _function_chunk_type(node: FuncNode, *, is_method: bool) -> ChunkType:
    """Map an AST node to the correct `ChunkType`."""
    if isinstance(node, ast.AsyncFunctionDef):
        return ChunkType.ASYNC_METHOD if is_method else ChunkType.ASYNC_FUNCTION
    return ChunkType.METHOD if is_method else ChunkType.FUNCTION


def _build_chunk(
    node: ast.AST,
    *,
    source: str,
    file_path: Path,
    chunk_type: ChunkType,
    name: str,
    qualified_name: str,
    parent: str | None,
) -> CodeChunk | None:
    """Construct a `CodeChunk` from an AST node, returning None if extraction fails."""
    segment = ast.get_source_segment(source, node)
    if segment is None or not segment.strip():
        return None

    start_line = getattr(node, "lineno", None)
    end_line = getattr(node, "end_lineno", None)
    if start_line is None or end_line is None:
        return None

    docstring: str | None = None
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        docstring = ast.get_docstring(node)

    return CodeChunk(
        chunk_type=chunk_type,
        name=name,
        qualified_name=qualified_name,
        source=segment,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        docstring=docstring,
        parent=parent,
    )


def _iter_methods(class_node: ast.ClassDef) -> Iterable[FuncNode]:
    """Yield only direct method definitions inside a class body."""
    for child in class_node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child


def parse_source(source: str, file_path: Path) -> list[CodeChunk]:
    """Parse a Python source string into a flat list of `CodeChunk`s."""
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return []

    chunks: list[CodeChunk] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk = _build_chunk(
                node,
                source=source,
                file_path=file_path,
                chunk_type=_function_chunk_type(node, is_method=False),
                name=node.name,
                qualified_name=node.name,
                parent=None,
            )
            if chunk is not None:
                chunks.append(chunk)

        elif isinstance(node, ast.ClassDef):
            class_chunk = _build_chunk(
                node,
                source=source,
                file_path=file_path,
                chunk_type=ChunkType.CLASS,
                name=node.name,
                qualified_name=node.name,
                parent=None,
            )
            if class_chunk is not None:
                chunks.append(class_chunk)

            for method in _iter_methods(node):
                method_chunk = _build_chunk(
                    method,
                    source=source,
                    file_path=file_path,
                    chunk_type=_function_chunk_type(method, is_method=True),
                    name=method.name,
                    qualified_name=f"{node.name}.{method.name}",
                    parent=node.name,
                )
                if method_chunk is not None:
                    chunks.append(method_chunk)

    return chunks


def parse_file(file_path: Path) -> list[CodeChunk]:
    """Read a `.py` file from disk and return its chunks."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Source file not found: {file_path}")
    if file_path.suffix != ".py":
        raise ValueError(f"Only .py files are supported: {file_path}")

    source = file_path.read_text(encoding="utf-8")
    return parse_source(source, file_path)


def parse_directory(
    directory: Path,
    *,
    exclude_dirs: frozenset[str] = frozenset(
        {".venv", "venv", "__pycache__", ".git", "node_modules"}
    ),
) -> list[CodeChunk]:
    """Recursively parse every `.py` file under `directory`."""
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    all_chunks: list[CodeChunk] = []
    for path in directory.rglob("*.py"):
        if any(part in exclude_dirs for part in path.parts):
            continue
        all_chunks.extend(parse_file(path))
    return all_chunks