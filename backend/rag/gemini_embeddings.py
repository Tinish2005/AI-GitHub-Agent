"""
Gemini-backed embedding backend with throttling + retry.

Avoids torch entirely (torch has no Python 3.14 support). Respects the
free-tier rate limit by embedding one item at a time with a small delay
and retrying automatically on 429 quota errors.
"""

from __future__ import annotations

import time


CANDIDATE_MODELS = [
    "models/text-embedding-004",
    "models/embedding-001",
    "models/gemini-embedding-001",
]


class GeminiEmbeddingBackend:
    """Embeds text via the Gemini embedding API, one item at a time."""

    def __init__(self, api_key: str, *, delay_seconds: float = 0.7) -> None:
        if not api_key:
            raise ValueError("Gemini API key must not be empty.")
        self._api_key = api_key
        self._delay = delay_seconds
        self._genai = None
        self._model = None

    def _ensure(self):
        if self._genai is None:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            self._genai = genai

    def _pick_model(self, sample: str) -> str:
        last_err = None
        for name in CANDIDATE_MODELS:
            try:
                self._genai.embed_content(model=name, content=sample)
                return name
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"No usable Gemini embedding model found: {last_err}")

    def _embed_one(self, text: str, model: str):
        """Embed a single string with retry-on-429."""
        for attempt in range(6):
            try:
                res = self._genai.embed_content(model=model, content=text)
                return list(res["embedding"])
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "quota" in msg.lower():
                    wait = 12.0
                    print(f"    rate limited, waiting {wait}s (attempt {attempt + 1})...")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("Gemini embedding kept hitting the rate limit.")

    def encode(
        self,
        sentences,
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ):
        self._ensure()
        texts = list(sentences)
        if not texts:
            return []
        if self._model is None:
            self._model = self._pick_model(texts[0])

        out = []
        total = len(texts)
        for i, text in enumerate(texts, start=1):
            out.append(self._embed_one(text, self._model))
            if i % 20 == 0:
                print(f"    embedded {i}/{total}")
            time.sleep(self._delay)
        return out
