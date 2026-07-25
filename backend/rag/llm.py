"""
LLM client abstraction.

A tiny `Protocol` so any backend (OpenAI today, Anthropic / Gemini /
local later) can be plugged in. Comes with two concrete implementations:

    - `OpenAILLMClient`: real production client.
    - `EchoLLMClient`: deterministic fake for tests — no network, no
      API key needed.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Anything that can answer a prompt and identify its model."""

    @property
    def model(self) -> str:
        ...

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


class EchoLLMClient:
    """
    Deterministic fake LLM used in tests.

    Returns a structured echo of the prompts so tests can assert on
    what the pipeline assembled. Never makes a network call.
    """

    def __init__(self, model: str = "echo-test") -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return (
            f"[ECHO answer | model={self._model}]\n"
            f"SYSTEM: {system_prompt.strip()[:120]}\n"
            f"USER: {user_prompt.strip()[:120]}"
        )


class OpenAILLMClient:
    """
    Real OpenAI-backed implementation.

    The OpenAI SDK is imported lazily so unit tests that use the fake
    backend never need the library at import time.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_output_tokens: int = 600,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key must not be empty.")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._client = None  # type: ignore[assignment]

    @property
    def model(self) -> str:
        return self._model

    def _ensure_client(self):  # type: ignore[no-untyped-def]
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_output_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return content.strip()