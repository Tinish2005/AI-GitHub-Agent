"""Unit tests for `backend.indexing.models`."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.indexing.models import ChunkType, CodeChunk


def _make_chunk(**overrides: object) -> CodeChunk:
    """Helper: build a valid chunk with optional field overrides."""
    defaults: dict[str, object] = {
        "chunk_type": ChunkType.FUNCTION,
        "name": "foo",
        "qualified_name": "foo",
        "source": "def foo():\n    return 1\n",
        "file_path": Path("sample.py"),
        "start_line": 1,
        "end_line": 2,
    }
    defaults.update(overrides)
    return CodeChunk(**defaults)  # type: ignore[arg-type]


def test_valid_chunk_is_created() -> None:
    chunk = _make_chunk()
    assert chunk.name == "foo"
    assert chunk.chunk_type == ChunkType.FUNCTION
    assert chunk.language == "python"
    assert chunk.line_count == 2


def test_line_count_is_inclusive() -> None:
    chunk = _make_chunk(start_line=10, end_line=20)
    assert chunk.line_count == 11


def test_end_line_before_start_line_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _make_chunk(start_line=5, end_line=3)


def test_empty_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _make_chunk(name="")


def test_chunk_is_frozen() -> None:
    chunk = _make_chunk()
    with pytest.raises(ValidationError):
        chunk.name = "bar"  # type: ignore[misc]


def test_method_chunk_has_parent() -> None:
    chunk = _make_chunk(
        chunk_type=ChunkType.METHOD,
        name="bar",
        qualified_name="Foo.bar",
        parent="Foo",
    )
    assert chunk.parent == "Foo"
    assert chunk.qualified_name == "Foo.bar"