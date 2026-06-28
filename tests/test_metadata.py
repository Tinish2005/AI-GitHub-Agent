"""Unit tests for `backend.indexing.metadata`."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from backend.indexing.ast_parser import parse_file, parse_source
from backend.indexing.metadata import (
    extract_metadata,
    extract_metadata_for_chunks,
)
from backend.indexing.models import ChunkMetadata


SAMPLE_SOURCE = '''\
"""Sample module."""

import os
from pathlib import Path
from collections import defaultdict


def greet(name: str) -> str:
    """Say hi."""
    return f"Hello, {name.upper()}"


class Calculator:
    """A tiny calculator."""

    def add(self, a: int, b: int) -> int:
        result = sum([a, b])
        return result
'''


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.py"
    f.write_text(SAMPLE_SOURCE, encoding="utf-8")
    return f


def test_extract_metadata_returns_chunk_metadata(sample_file: Path) -> None:
    chunks = parse_file(sample_file)
    meta = extract_metadata(chunks[0])
    assert isinstance(meta, ChunkMetadata)
    assert meta.qualified_name == chunks[0].qualified_name


def test_content_hash_is_lowercase_sha256(sample_file: Path) -> None:
    chunks = parse_file(sample_file)
    meta = extract_metadata(chunks[0])

    expected = hashlib.sha256(chunks[0].source.encode("utf-8")).hexdigest()
    assert meta.content_hash == expected
    assert meta.content_hash == meta.content_hash.lower()
    assert len(meta.content_hash) == 64


def test_imports_are_extracted_and_sorted(sample_file: Path) -> None:
    chunks = parse_file(sample_file)
    meta = extract_metadata(chunks[0])

    assert meta.imports == ("collections", "os", "pathlib")
    # tuple, not list (immutable)
    assert isinstance(meta.imports, tuple)


def test_calls_are_extracted_for_function(sample_file: Path) -> None:
    chunks = parse_file(sample_file)
    greet = next(c for c in chunks if c.qualified_name == "greet")
    meta = extract_metadata(greet)
    assert "upper" in meta.calls


def test_calls_are_extracted_for_method(sample_file: Path) -> None:
    chunks = parse_file(sample_file)
    add_method = next(c for c in chunks if c.qualified_name == "Calculator.add")
    meta = extract_metadata(add_method)
    assert "sum" in meta.calls


def test_module_path_with_project_root(tmp_path: Path) -> None:
    pkg = tmp_path / "backend" / "indexing"
    pkg.mkdir(parents=True)
    f = pkg / "ast_parser.py"
    f.write_text("def x():\n    return 1\n", encoding="utf-8")

    chunks = parse_file(f)
    meta = extract_metadata(chunks[0], project_root=tmp_path)
    assert meta.module_path == "backend.indexing.ast_parser"


def test_module_path_strips_init(tmp_path: Path) -> None:
    pkg = tmp_path / "backend" / "indexing"
    pkg.mkdir(parents=True)
    init = pkg / "__init__.py"
    init.write_text('"""pkg."""\n\ndef x():\n    return 1\n', encoding="utf-8")

    chunks = parse_file(init)
    meta = extract_metadata(chunks[0], project_root=tmp_path)
    assert meta.module_path == "backend.indexing"


def test_module_path_falls_back_without_root(sample_file: Path) -> None:
    chunks = parse_file(sample_file)
    meta = extract_metadata(chunks[0])  # no project_root
    assert meta.module_path == "sample"


def test_module_path_falls_back_when_unrelated(tmp_path: Path) -> None:
    f = tmp_path / "alpha.py"
    f.write_text("def x():\n    return 1\n", encoding="utf-8")
    chunks = parse_file(f)

    other_root = tmp_path / "elsewhere"
    other_root.mkdir()
    meta = extract_metadata(chunks[0], project_root=other_root)
    assert meta.module_path == "alpha"


def test_missing_file_yields_empty_imports() -> None:
    # Build a chunk pointing at a file that does not exist.
    chunks = parse_source(
        "def foo():\n    return 1\n",
        Path("does_not_exist.py"),
    )
    meta = extract_metadata(chunks[0])
    assert meta.imports == ()


def test_extract_metadata_for_chunks_caches_imports(
    sample_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Imports should be parsed once per file, not once per chunk."""
    from backend.indexing import metadata as metadata_module

    calls: list[Path] = []
    original = metadata_module._extract_file_imports

    def spy(path: Path) -> tuple[str, ...]:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(metadata_module, "_extract_file_imports", spy)

    chunks = parse_file(sample_file)
    assert len(chunks) >= 3  # function + class + method
    metas = extract_metadata_for_chunks(chunks)

    assert len(metas) == len(chunks)
    # Only ONE call to _extract_file_imports for all chunks of the same file.
    assert calls.count(sample_file) == 1


def test_metadata_is_frozen(sample_file: Path) -> None:
    from pydantic import ValidationError

    chunks = parse_file(sample_file)
    meta = extract_metadata(chunks[0])
    with pytest.raises(ValidationError):
        meta.module_path = "tampered"  # type: ignore[misc]


def test_invalid_hash_is_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChunkMetadata(
            qualified_name="foo",
            file_path=Path("x.py"),
            module_path="x",
            content_hash="ZZZZ" * 16,  # not hex
            imports=(),
            calls=(),
        )


def test_uppercase_hash_is_rejected() -> None:
    from pydantic import ValidationError

    upper = "A" * 64
    with pytest.raises(ValidationError):
        ChunkMetadata(
            qualified_name="foo",
            file_path=Path("x.py"),
            module_path="x",
            content_hash=upper,
            imports=(),
            calls=(),
        )