"""One place that decides which embedding model the app uses."""

from __future__ import annotations

from backend.indexing.embeddings import EmbeddingService


def build_embedding_service(settings) -> EmbeddingService:
    """Gemini embeddings when a key is set (model pinned if configured), else local."""
    key = settings.gemini_api_key.get_secret_value()
    if key:
        from backend.rag.gemini_embeddings import GeminiEmbeddingBackend

        model = (getattr(settings, "gemini_embedding_model", "") or "").strip() or None
        return EmbeddingService(backend=GeminiEmbeddingBackend(api_key=key, model=model))
    return EmbeddingService()
