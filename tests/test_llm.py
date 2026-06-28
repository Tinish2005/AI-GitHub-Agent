"""Unit tests for `backend.rag.llm`."""

from __future__ import annotations

import pytest

from backend.rag.llm import EchoLLMClient, OpenAILLMClient


def test_echo_client_returns_deterministic_text() -> None:
    client = EchoLLMClient()
    out = client.complete("be helpful", "hello")
    assert "ECHO answer" in out
    assert "be helpful" in out
    assert "hello" in out


def test_echo_client_model_name_default() -> None:
    assert EchoLLMClient().model == "echo-test"


def test_echo_client_model_name_override() -> None:
    assert EchoLLMClient(model="m1").model == "m1"


def test_openai_client_rejects_empty_key() -> None:
    with pytest.raises(ValueError):
        OpenAILLMClient(api_key="")


def test_openai_client_stores_model_name() -> None:
    c = OpenAILLMClient(api_key="sk-fake", model="gpt-4o-mini")
    assert c.model == "gpt-4o-mini"