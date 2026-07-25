"""
Gemini-backed LLM client.

Implements the same LLMClient protocol as OpenAILLMClient and
EchoLLMClient, so the rest of the agent does not care which backend
is in use. The SDK is imported lazily so tests that use a fake backend
never need the library installed.
"""

from __future__ import annotations


class GeminiLLMClient:
    """Real Google Gemini implementation of the LLMClient protocol."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
    ) -> None:
        if not api_key:
            raise ValueError("Gemini API key must not be empty.")
        self._api_key = api_key
        self._model_name = model
        self._temperature = temperature
        self._client = None

    @property
    def model(self) -> str:
        return self._model_name

    def _ensure_client(self):
        if self._client is None:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self._model_name)
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        client = self._ensure_client()
        prompt = f"{system_prompt}\n\n{user_prompt}"
        response = client.generate_content(
            prompt,
            generation_config={"temperature": self._temperature},
        )
        text = getattr(response, "text", "") or ""
        return text.strip()
