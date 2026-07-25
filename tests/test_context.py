"""Unit tests for `backend.rag.context`."""

from __future__ import annotations

import pytest

from backend.indexing.vector_store import QueryResult
from backend.rag.context import assemble_context


def _result(chunk_id: str, distance: float, doc: str = "def foo():\n    pass") -> QueryResult:
    return QueryResult(
        chunk_id=chunk_id,
        document=doc,
        metadata={
            "qualified_name": f"mod.{chunk_id}",
            "file_path": "mod.py",
            "start_line": "1",
            "end_line": "2",
        },
        distance=distance,
    )


def test_assemble_orders_by_distance() -> None:
    out = assemble_context(
        [
            _result("c", 0.9),
            _result("a", 0.1),
            _result("b", 0.5),
        ]
    )
    ids = [r.chunk_id for r in out.kept]
    assert ids == ["a", "b", "c"]


def test_assemble_dedupes_by_chunk_id() -> None:
    out = assemble_context(
        [
            _result("a", 0.1),
            _result("a", 0.2),  # duplicate id
            _result("b", 0.3),
        ]
    )
    assert [r.chunk_id for r in out.kept] == ["a", "b"]


def test_assemble_respects_budget() -> None:
    big_doc = "x" * 1000
    out = assemble_context(
        [_result("a", 0.1, big_doc), _result("b", 0.2, big_doc), _result("c", 0.3, big_doc)],
        budget_chars=1500,
    )
    assert len(out.kept) >= 1
    assert out.used_chars <= 1500 + 200  # tiny slack for headers/fencing


def test_assemble_keeps_at_least_one_when_budget_tiny() -> None:
    """Even with a tiny budget the first (best) chunk should be kept."""
    out = assemble_context([_result("a", 0.1, "x" * 5000)], budget_chars=10)
    assert len(out.kept) == 1


def test_assemble_handles_empty_input() -> None:
    out = assemble_context([])
    assert out.text == ""
    assert out.kept == ()
    assert out.used_chars == 0


def test_assemble_rejects_zero_budget() -> None:
    with pytest.raises(ValueError):
        assemble_context([], budget_chars=0)


def test_render_includes_citation_numbers() -> None:
    out = assemble_context([_result("a", 0.1), _result("b", 0.2)])
    assert "[1] mod.a" in out.text
    assert "[2] mod.b" in out.text