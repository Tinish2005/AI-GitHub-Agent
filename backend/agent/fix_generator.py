"""
FixGenerator service.

Takes a goal + retrieved context and asks an LLM to produce a unified
diff plus a short explanation, returned as a JSON object. We then
validate the diff via the `unidiff` library so downstream steps
(validation, PR creation) can trust the shape of the payload.

Design notes:
    - The LLM must return a JSON object with `explanation`, `diff`, and
      an optional `confidence`. Anything else is a hard error, exposed
      via `FixProposal.is_valid=False` and a filled `validation_error`.
    - `unidiff` gives us per-file stats for free.
    - We NEVER apply the diff here. Applying and testing belongs to
      a later stage. This module is purely "propose and describe".
"""

from __future__ import annotations

import json

from backend.agent.fix_models import FixHunk, FixProposal
from backend.rag.llm import LLMClient


FIX_SYSTEM_PROMPT: str = (
    "You are an expert software engineer producing minimal, well-scoped code fixes. "
    "Given a goal and code context, output a JSON object with EXACTLY these keys: "
    "'explanation' (string, plain English), 'diff' (string, a valid unified diff), "
    "and 'confidence' (float, 0.0 to 1.0). Return ONLY the JSON - no prose."
)


class FixGenerator:
    """Produces FixProposal objects from a goal and code context."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def propose(self, goal: str, context: str) -> FixProposal:
        """Ask the LLM for a fix, parse it, validate the diff."""
        if not goal or not goal.strip():
            raise ValueError("Goal must not be empty.")
        goal = goal.strip()

        user_prompt = self._build_user_prompt(goal, context)
        response = self.llm.complete(FIX_SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_json(response)

        if parsed is None:
            return FixProposal(
                goal=goal,
                explanation="(LLM did not return valid JSON)",
                diff=response.strip() or "(empty)",
                hunks=(),
                model=self.llm.model,
                confidence=0.0,
                is_valid=False,
                validation_error="Response was not valid JSON.",
            )

        explanation = str(parsed.get("explanation", "")).strip()
        diff = str(parsed.get("diff", "")).strip()
        raw_conf = parsed.get("confidence", 0.5)
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = 0.5

        if not explanation or not diff:
            return FixProposal(
                goal=goal,
                explanation=explanation or "(missing)",
                diff=diff or "(missing)",
                hunks=(),
                model=self.llm.model,
                confidence=confidence,
                is_valid=False,
                validation_error="Response was missing 'explanation' or 'diff'.",
            )

        hunks, is_valid, err = self._validate_diff(diff)

        return FixProposal(
            goal=goal,
            explanation=explanation,
            diff=diff,
            hunks=hunks,
            model=self.llm.model,
            confidence=confidence,
            is_valid=is_valid,
            validation_error=err,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_prompt(goal: str, context: str) -> str:
        return (
            f"Goal: {goal}\n\n"
            f"Code context (retrieved chunks):\n{context if context.strip() else '(none)'}\n\n"
            "Produce the JSON object as described in the system prompt."
        )

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Try to extract a JSON object from the LLM response."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _validate_diff(diff_text: str) -> tuple:
        """Return (hunks, is_valid, error_message)."""
        try:
            from unidiff import PatchSet
        except ImportError:
            return (), False, "unidiff not installed - run pip install -r requirements.txt"

        try:
            patch = PatchSet(diff_text)
        except Exception as exc:
            return (), False, f"Failed to parse diff: {type(exc).__name__}: {exc}"

        if len(patch) == 0:
            return (), False, "Diff contains no file changes."

        hunks: list = []
        for patched_file in patch:
            path = patched_file.path or patched_file.source_file or "unknown"
            hunks.append(
                FixHunk(
                    file_path=str(path),
                    is_new_file=bool(patched_file.is_added_file),
                    is_deleted_file=bool(patched_file.is_removed_file),
                    added_lines=int(patched_file.added),
                    removed_lines=int(patched_file.removed),
                )
            )
        return tuple(hunks), True, ""