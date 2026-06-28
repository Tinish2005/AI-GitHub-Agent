"""
Data models for the indexing pipeline.

A `CodeChunk` is the atomic unit produced by the parser: one function or one
class extracted from a source file. `ChunkMetadata` adds the analysis layer
on top of a chunk — imports, calls, hash, and module path — needed by later
loops (repo graph, dedup, change detection).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChunkType(str, Enum):
    """Kind of code element captured in a chunk."""

    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    CLASS = "class"
    METHOD = "method"
    ASYNC_METHOD = "async_method"


class CodeChunk(BaseModel):
    """
    A single semantically-meaningful slice of source code.

    Each chunk is what later loops will embed, index, and retrieve from.
    Keeping this model strict and immutable prevents corruption downstream.
    """

    model_config = {"frozen": True}

    chunk_type: ChunkType = Field(description="Kind of code element.")
    name: str = Field(min_length=1, description="Name of the element.")
    qualified_name: str = Field(min_length=1, description="Dotted path within the module.")
    source: str = Field(min_length=1, description="Verbatim source code of the chunk.")
    file_path: Path = Field(description="Path to the source file the chunk came from.")
    start_line: int = Field(ge=1, description="1-indexed inclusive start line.")
    end_line: int = Field(ge=1, description="1-indexed inclusive end line.")
    docstring: str | None = Field(default=None, description="Docstring if present.")
    parent: str | None = Field(
        default=None,
        description="Name of the enclosing class for methods; None for top-level items.",
    )
    language: Literal["python"] = Field(
        default="python",
        description="Source language. Pinned to Python for Loop 2.",
    )

    @field_validator("end_line")
    @classmethod
    def _end_after_start(cls, v: int, info) -> int:
        """end_line must be >= start_line."""
        start = info.data.get("start_line")
        if start is not None and v < start:
            raise ValueError(f"end_line ({v}) must be >= start_line ({start})")
        return v

    @property
    def line_count(self) -> int:
        """Number of source lines covered by the chunk (inclusive)."""
        return self.end_line - self.start_line + 1


class ChunkMetadata(BaseModel):
    """
    Analysis-layer metadata attached to a `CodeChunk`.

    Produced by `backend.indexing.metadata.extract_metadata`. Kept separate
    from `CodeChunk` so the parser output stays clean and immutable, while
    metadata can evolve independently as later loops add fields.
    """

    model_config = {"frozen": True}

    qualified_name: str = Field(
        min_length=1,
        description="qualified_name of the chunk this metadata belongs to.",
    )
    file_path: Path = Field(description="Path to the source file the chunk came from.")
    module_path: str = Field(
        min_length=1,
        description="Dotted module path derived from file_path (e.g. 'backend.indexing.ast_parser').",
    )
    content_hash: str = Field(
        min_length=64,
        max_length=64,
        description="Lowercase SHA-256 hex digest of the chunk's source code.",
    )
    imports: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Top-level imports visible in the file (deduped, alphabetically sorted).",
    )
    calls: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Function/method call names found inside the chunk (deduped, sorted).",
    )

    @field_validator("content_hash")
    @classmethod
    def _is_hex(cls, v: str) -> str:
        """content_hash must be a valid lowercase hex string."""
        try:
            int(v, 16)
        except ValueError as e:
            raise ValueError(f"content_hash must be valid hex: {v}") from e
        if v != v.lower():
            raise ValueError("content_hash must be lowercase")
        return v