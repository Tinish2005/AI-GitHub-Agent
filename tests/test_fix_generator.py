"""Unit tests for FixGenerator."""

from __future__ import annotations

import json

import pytest

from backend.agent.fix_generator import FixGenerator


class _StaticLLM:
    def __init__(self, response: str, model: str = "static-fake") -> None:
        self._response = response
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


# A well-formed unified diff. Hunk header @@ -1,1 +1,1 @@ means:
#   original: starting line 1, 1 line;  new: starting line 1, 1 line.
VALID_DIFF = (
    "--- a/src/app.py\n"
    "+++ b/src/app.py\n"
    "@@ -1,1 +1,1 @@\n"
    "-return a - b\n"
    "+return a + b\n"
)


def _valid_response() -> str:
    return json.dumps({
        "explanation": "Fixed a subtraction bug in add().",
        "diff": VALID_DIFF,
        "confidence": 0.9,
    })


def test_fix_generator_parses_valid_response() -> None:
    gen = FixGenerator(llm=_StaticLLM(_valid_response()))
    proposal = gen.propose("Fix the add bug", "some context")
    assert proposal.is_valid is True, f"unexpected error: {proposal.validation_error}"
    assert proposal.confidence == 0.9
    assert proposal.model == "static-fake"
    assert proposal.files_changed == 1
    assert proposal.total_added >= 1
    assert proposal.total_removed >= 1


def test_fix_generator_handles_json_wrapped_in_prose() -> None:
    wrapped = "Sure!\n\n" + _valid_response() + "\n\nDone."
    gen = FixGenerator(llm=_StaticLLM(wrapped))
    proposal = gen.propose("Fix the add bug", "")
    assert proposal.is_valid is True


def test_fix_generator_marks_invalid_when_not_json() -> None:
    gen = FixGenerator(llm=_StaticLLM("this is not json at all"))
    proposal = gen.propose("Fix stuff", "")
    assert proposal.is_valid is False
    assert "not valid JSON" in proposal.validation_error


def test_fix_generator_marks_invalid_when_missing_fields() -> None:
    bad = json.dumps({"explanation": "only this"})
    gen = FixGenerator(llm=_StaticLLM(bad))
    proposal = gen.propose("Fix stuff", "")
    assert proposal.is_valid is False
    assert "missing" in proposal.validation_error.lower()


def test_fix_generator_marks_invalid_on_bad_diff() -> None:
    bad = json.dumps({
        "explanation": "e",
        "diff": "this is not a diff at all",
        "confidence": 0.5,
    })
    gen = FixGenerator(llm=_StaticLLM(bad))
    proposal = gen.propose("Fix stuff", "")
    assert proposal.is_valid is False


def test_fix_generator_defaults_confidence_when_missing() -> None:
    bad = json.dumps({"explanation": "e", "diff": VALID_DIFF})
    gen = FixGenerator(llm=_StaticLLM(bad))
    proposal = gen.propose("Fix stuff", "")
    assert 0.0 <= proposal.confidence <= 1.0


def test_fix_generator_rejects_empty_goal() -> None:
    gen = FixGenerator(llm=_StaticLLM(_valid_response()))
    with pytest.raises(ValueError):
        gen.propose("   ", "context")