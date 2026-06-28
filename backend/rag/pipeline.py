"""
RAG pipeline orchestrator.

Takes a question → retrieves top-k chunks from the vector store →
assembles context → asks the LLM → returns a structured `Answer`
with the sources it used.
"""

from __future__ import annotations

from backend.indexing.vector_store import QueryResult, VectorStore
from backend.rag.context import (
    DEFAULT_CONTEXT_BUDGET_CHARS,
    AssembledContext,
    assemble_context,
)
from backend.rag.llm import LLMClient
from backend.rag.models import Answer, Source


SYSTEM_PROMPT: str = (
    "You are an expert software engineer answering questions about a Python "
    "codebase. Use ONLY the provided context to answer. Cite sources by their "
    "bracketed number, e.g. [1], [2]. If the context does not contain the "
    "answer, say so plainly instead of inventing details."
)


class RAGPipeline:
    """Compose a `VectorStore` and an `LLMClient` into a Q&A system."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm: LLMClient,
        *,
        top_k: int = 5,
        budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        self.budget_chars = budget_chars

    def ask(self, question: str) -> Answer:
        """Run the full retrieve → assemble → answer flow."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        results = self.vector_store.query(question, top_k=self.top_k)
        context = assemble_context(results, budget_chars=self.budget_chars)
        user_prompt = self._build_user_prompt(question, context)
        answer_text = self.llm.complete(SYSTEM_PROMPT, user_prompt)

        sources = tuple(_to_source(r) for r in context.kept)
        return Answer(
            question=question,
            answer=answer_text,
            sources=sources,
            model=self.llm.model,
            used_context_chars=context.used_chars,
        )

    @staticmethod
    def _build_user_prompt(question: str, context: AssembledContext) -> str:
        if not context.text:
            return (
                f"Question: {question}\n\n"
                "Context: (no relevant code chunks were retrieved)\n"
            )
        return (
            f"Question: {question}\n\n"
            f"Context:\n{context.text}\n"
        )


def _to_source(r: QueryResult) -> Source:
    """Convert a `QueryResult` to a public `Source` model."""
    meta = r.metadata or {}
    return Source(
        chunk_id=r.chunk_id,
        qualified_name=meta.get("qualified_name", "<unknown>"),
        file_path=meta.get("file_path", "<unknown>"),
        start_line=int(meta.get("start_line", 1) or 1),
        end_line=int(meta.get("end_line", 1) or 1),
        distance=float(r.distance),
    )