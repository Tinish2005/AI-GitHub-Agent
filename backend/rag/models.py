"""
Data models for the RAG pipeline.

Kept minimal and immutable so they can be safely passed across the
pipeline (retrieval → assembly → LLM → API response) without anyone
accidentally mutating them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single source citation attached to an answer."""

    model_config = {"frozen": True}

    chunk_id: str = Field(min_length=1, description="Content hash of the source chunk.")
    qualified_name: str = Field(min_length=1, description="Dotted name of the chunk.")
    file_path: str = Field(min_length=1, description="Source file path.")
    start_line: int = Field(ge=1, description="1-indexed start line.")
    end_line: int = Field(ge=1, description="1-indexed end line.")
    distance: float = Field(ge=0.0, description="Vector distance (lower is closer).")


class Answer(BaseModel):
    """A complete RAG answer with the sources used to produce it."""

    model_config = {"frozen": True}

    question: str = Field(min_length=1, description="The user's question.")
    answer: str = Field(min_length=1, description="The LLM's answer text.")
    sources: tuple[Source, ...] = Field(
        default_factory=tuple,
        description="Source chunks the LLM was given as context.",
    )
    model: str = Field(min_length=1, description="LLM model identifier used.")
    used_context_chars: int = Field(
        ge=0,
        description="Total characters of context that were sent to the LLM.",
    )

    @property
    def source_count(self) -> int:
        """How many sources were used."""
        return len(self.sources)