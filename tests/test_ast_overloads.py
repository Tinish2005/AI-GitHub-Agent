"""Loop 7b tests: @overload type-hint stubs are not indexed as separate chunks (OFFLINE)."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.indexing.ast_parser import parse_source
from tests.test_index_safety import _store

PATH = Path("m.py")

MODULE = '''\
from typing import overload

@overload
def f(a: int) -> int: ...

@overload
def f(a: str) -> str: ...

def f(a):
    """The real one."""
    return a

def other():
    return 1
'''

CLASS = '''\
import typing

class Box:
    @typing.overload
    def get(self, a: int) -> int: ...

    @typing.overload
    def get(self, a: str) -> str: ...

    def get(self, a):
        return a

    def put(self, a):
        return a
'''


def _names(chunks) -> list:
    return [c.qualified_name for c in chunks]


def test_stubs_are_dropped_when_the_real_function_exists() -> None:
    chunks = parse_source(MODULE, PATH)
    assert _names(chunks) == ["f", "other"]
    assert "The real one." in chunks[0].source and "..." not in chunks[0].source


@pytest.mark.parametrize("decorator", ["overload", "typing.overload", "t.overload", "typing_extensions.overload"])
def test_every_common_way_of_writing_overload_is_recognised(decorator: str) -> None:
    src = f"@{decorator}\ndef f(a: int) -> int: ...\n\ndef f(a):\n    return a\n"
    assert _names(parse_source(src, PATH)) == ["f"]


def test_method_stubs_are_dropped_but_the_class_chunk_still_shows_them() -> None:
    chunks = parse_source(CLASS, PATH)
    assert _names(chunks) == ["Box", "Box.get", "Box.put"]
    assert chunks[0].source.count("typing.overload") == 2
    assert "typing.overload" not in chunks[1].source and "return a" in chunks[1].source


def test_async_stubs_are_dropped_too() -> None:
    src = "@overload\nasync def f(a: int) -> int: ...\n\nasync def f(a):\n    return a\n"
    chunks = parse_source(src, PATH)
    assert len(chunks) == 1 and chunks[0].chunk_type.value == "async_function"


def test_a_name_with_only_stubs_keeps_them() -> None:
    src = "@overload\ndef f(a: int) -> int: ...\n\n@overload\ndef f(a: str) -> str: ...\n"
    assert _names(parse_source(src, PATH)) == ["f", "f"]
    protocol = "class P:\n    @overload\n    def m(self, a: int) -> int: ...\n"
    assert _names(parse_source(protocol, PATH)) == ["P", "P.m"]


def test_other_decorators_and_lookalike_names_are_untouched() -> None:
    src = (
        "@staticmethod\ndef a():\n    return 1\n\n"
        "@overloaded\ndef b():\n    return 2\n\n"
        "@my.overload_helper\ndef c():\n    return 3\n\n"
        "@property\ndef d(self):\n    return 4\n"
    )
    assert _names(parse_source(src, PATH)) == ["a", "b", "c", "d"]


def test_stubs_in_one_scope_do_not_hide_a_function_in_another() -> None:
    src = (
        "@overload\ndef get(a: int) -> int: ...\n\n"
        "class K:\n    def get(self):\n        return 1\n"
    )
    assert _names(parse_source(src, PATH)) == ["get", "K", "K.get"]


def test_the_indexed_names_are_unique_and_search_finds_the_real_function(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunks = parse_source(MODULE.replace("The real one.", "alpha one"), PATH)
    store.add_chunks(chunks, extra_metadata={"repo": "o/r"})
    names = [m["qualified_name"] for m in store.all_metadata()]
    assert names == ["f", "other"] and store.count() == len(chunks)
    top = store.query("alpha", top_k=3)
    assert [r.metadata["qualified_name"] for r in top][0] == "f"
    assert all("..." not in r.document for r in top if r.metadata["qualified_name"] == "f")


def test_real_requests_source_has_no_repeated_names_after_the_fix() -> None:
    requests = pytest.importorskip("requests")
    package = Path(requests.__file__).parent
    models = parse_source((package / "models.py").read_text(encoding="utf-8"), Path("src/requests/models.py"))
    names = _names(models)
    assert len(names) == len(set(names))
    encode = [c for c in models if c.qualified_name == "RequestEncodingMixin._encode_params"]
    assert len(encode) == 1 and "urlencode" in encode[0].source
