"""Unit tests for `backend.indexing.ast_parser`."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.indexing.ast_parser import parse_directory, parse_file, parse_source
from backend.indexing.models import ChunkType


SAMPLE_SOURCE = '''\
"""Sample module."""

def top_level_function(x: int) -> int:
    """Doubles x."""
    return x * 2


async def async_top_level() -> None:
    """An async top-level function."""
    return None


class Greeter:
    """A friendly greeter."""

    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        """Return a greeting."""
        return f"Hello, {self.name}"

    async def async_greet(self) -> str:
        return f"Hello async, {self.name}"
'''


def test_parse_source_finds_top_level_function() -> None:
    chunks = parse_source(SAMPLE_SOURCE, Path("sample.py"))
    funcs = [c for c in chunks if c.qualified_name == "top_level_function"]
    assert len(funcs) == 1
    assert funcs[0].chunk_type == ChunkType.FUNCTION
    assert funcs[0].docstring == "Doubles x."
    assert funcs[0].parent is None


def test_parse_source_finds_async_top_level() -> None:
    chunks = parse_source(SAMPLE_SOURCE, Path("sample.py"))
    asyncs = [c for c in chunks if c.qualified_name == "async_top_level"]
    assert len(asyncs) == 1
    assert asyncs[0].chunk_type == ChunkType.ASYNC_FUNCTION


def test_parse_source_finds_class_and_methods() -> None:
    chunks = parse_source(SAMPLE_SOURCE, Path("sample.py"))
    qnames = {c.qualified_name: c for c in chunks}

    assert "Greeter" in qnames
    assert qnames["Greeter"].chunk_type == ChunkType.CLASS
    assert qnames["Greeter"].docstring == "A friendly greeter."

    assert "Greeter.__init__" in qnames
    assert qnames["Greeter.__init__"].chunk_type == ChunkType.METHOD
    assert qnames["Greeter.__init__"].parent == "Greeter"

    assert "Greeter.greet" in qnames
    assert qnames["Greeter.greet"].chunk_type == ChunkType.METHOD

    assert "Greeter.async_greet" in qnames
    assert qnames["Greeter.async_greet"].chunk_type == ChunkType.ASYNC_METHOD


def test_parse_source_total_chunk_count() -> None:
    chunks = parse_source(SAMPLE_SOURCE, Path("sample.py"))
    assert len(chunks) == 6


def test_parse_source_source_is_verbatim() -> None:
    chunks = parse_source(SAMPLE_SOURCE, Path("sample.py"))
    fn = next(c for c in chunks if c.qualified_name == "top_level_function")
    assert "def top_level_function(x: int) -> int:" in fn.source
    assert "return x * 2" in fn.source


def test_parse_source_handles_syntax_error() -> None:
    bad_source = "def broken(:\n    pass\n"
    assert parse_source(bad_source, Path("bad.py")) == []


def test_parse_source_handles_empty_file() -> None:
    assert parse_source("", Path("empty.py")) == []


def test_parse_file_reads_from_disk(tmp_path: Path) -> None:
    file = tmp_path / "sample.py"
    file.write_text(SAMPLE_SOURCE, encoding="utf-8")
    chunks = parse_file(file)
    assert len(chunks) == 6
    assert all(c.file_path == file for c in chunks)


def test_parse_file_rejects_non_python(tmp_path: Path) -> None:
    f = tmp_path / "note.txt"
    f.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_file(f)


def test_parse_file_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_file(tmp_path / "does_not_exist.py")


def test_parse_directory_walks_recursively(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "sub").mkdir()
    (tmp_path / "pkg" / "sub" / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")

    chunks = parse_directory(tmp_path)
    names = {c.name for c in chunks}
    assert names == {"alpha", "beta"}


def test_parse_directory_skips_excluded_dirs(tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "junk.py").write_text("def noise():\n    return 0\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "real.py").write_text("def real():\n    return 1\n", encoding="utf-8")

    chunks = parse_directory(tmp_path)
    names = {c.name for c in chunks}
    assert names == {"real"}


def test_parse_directory_rejects_non_directory(tmp_path: Path) -> None:
    f = tmp_path / "file.py"
    f.write_text("def x():\n    pass\n", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        parse_directory(f)