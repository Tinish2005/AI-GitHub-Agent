"""Unit tests for backend.rag.gemini_llm."""

from __future__ import annotations

import pytest

from backend.rag.gemini_llm import GeminiLLMClient


def test_gemini_client_rejects_empty_key() -> None:
    with pytest.raises(ValueError):
        GeminiLLMClient(api_key="")


def test_gemini_client_stores_model_name() -> None:
    client = GeminiLLMClient(api_key="fake-key", model="gemini-2.5-flash")
    assert client.model == "gemini-2.5-flash"


def test_gemini_client_default_model() -> None:
    client = GeminiLLMClient(api_key="fake-key")
    assert client.model == "gemini-2.5-flash"


def test_gemini_client_lazy_no_client_until_used() -> None:
    client = GeminiLLMClient(api_key="fake-key")
    assert client._client is None
