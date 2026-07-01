"""
Planners.

A Planner turns a user goal into a Plan. Two implementations ship in
Loop 9:

    - RuleBasedPlanner: deterministic, no LLM. Picks a plan template
      based on keywords in the goal. Fast, cheap, testable.

    - LLMPlanner: wraps an LLMClient and asks it to produce a plan in
      a strict JSON shape. Falls back to the rule-based planner if
      the LLM returns malformed output ("graceful degradation").
"""

from __future__ import annotations

import json
from typing import Protocol

from backend.agent.models import Plan, PlanStep, StepKind
from backend.rag.llm import LLMClient


class Planner(Protocol):
    """Anything that can turn a user goal into a Plan."""

    strategy: str

    def plan(self, goal: str) -> Plan:
        ...


PLANNER_SYSTEM_PROMPT: str = (
    "You are an expert software engineer planning multi-step work on a "
    "codebase. Given a user goal, output a JSON object with a 'steps' "
    "array. Each step must have: 'kind' (one of: retrieve, analyze, "
    "github_read, generate, validate, human_approval, draft_pr) and "
    "'description'. Return ONLY the JSON - no prose."
)


def _keywords(goal: str) -> set:
    """Return lowercased tokens for keyword matching."""
    return set(goal.lower().split())


class RuleBasedPlanner:
    """Deterministic planner that picks a template based on the goal."""

    strategy: str = "rule_based"

    def plan(self, goal: str) -> Plan:
        if not goal or not goal.strip():
            raise ValueError("Goal must not be empty.")

        words = _keywords(goal)
        template = self._pick_template(words)
        steps = tuple(
            PlanStep(id=i + 1, kind=kind, description=desc)
            for i, (kind, desc) in enumerate(template)
        )
        return Plan(goal=goal.strip(), steps=steps, strategy=self.strategy)

    @staticmethod
    def _pick_template(words: set) -> list:
        """Pick a plan template based on which category the goal falls into."""
        fix_words = {"fix", "bug", "broken", "error", "crash", "failing"}
        review_words = {"review", "audit", "explain", "understand"}
        pr_words = {"pr", "pull", "request", "propose", "submit"}
        github_words = {"issue", "issues", "github", "repo", "repository"}

        if words & fix_words:
            return [
                (StepKind.RETRIEVE, "Retrieve code chunks related to the reported bug."),
                (StepKind.GITHUB_READ, "Read the failing issue or PR discussion for context."),
                (StepKind.ANALYZE, "Diagnose the root cause from the retrieved context."),
                (StepKind.GENERATE, "Generate a proposed code fix as a diff."),
                (StepKind.VALIDATE, "Run tests + lint to validate the fix."),
                (StepKind.HUMAN_APPROVAL, "Wait for a human to review the diff."),
                (StepKind.DRAFT_PR, "Push the fix as a draft pull request."),
            ]
        if words & pr_words:
            return [
                (StepKind.RETRIEVE, "Retrieve the code area the change should touch."),
                (StepKind.ANALYZE, "Plan the change scope and impact."),
                (StepKind.GENERATE, "Generate the proposed change as a diff."),
                (StepKind.VALIDATE, "Run tests + lint on the proposed change."),
                (StepKind.HUMAN_APPROVAL, "Wait for human review of the diff."),
                (StepKind.DRAFT_PR, "Push a draft pull request."),
            ]
        if words & review_words:
            return [
                (StepKind.RETRIEVE, "Retrieve the code chunks referenced in the goal."),
                (StepKind.ANALYZE, "Summarize the retrieved code and highlight risks."),
            ]
        if words & github_words:
            return [
                (StepKind.GITHUB_READ, "Read the relevant GitHub items (issues / PRs / files)."),
                (StepKind.ANALYZE, "Summarize the findings for the user."),
            ]
        # Default: Q&A over the codebase.
        return [
            (StepKind.RETRIEVE, "Retrieve top-k relevant code chunks for the goal."),
            (StepKind.ANALYZE, "Answer the goal using the retrieved context."),
        ]


class LLMPlanner:
    """LLM-backed planner. Falls back to RuleBasedPlanner on bad JSON."""

    strategy: str = "llm"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self._fallback = RuleBasedPlanner()

    def plan(self, goal: str) -> Plan:
        if not goal or not goal.strip():
            raise ValueError("Goal must not be empty.")

        response = self.llm.complete(PLANNER_SYSTEM_PROMPT, f"Goal: {goal.strip()}")
        parsed = self._parse_json(response)
        if parsed is None:
            return self._fallback.plan(goal)

        raw_steps = parsed.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            return self._fallback.plan(goal)

        steps: list = []
        for i, raw in enumerate(raw_steps, start=1):
            if not isinstance(raw, dict):
                return self._fallback.plan(goal)
            kind = self._parse_kind(raw.get("kind"))
            desc = str(raw.get("description", "")).strip()
            if kind is None or not desc:
                return self._fallback.plan(goal)
            steps.append(PlanStep(id=i, kind=kind, description=desc))

        return Plan(goal=goal.strip(), steps=tuple(steps), strategy=self.strategy)

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Try to extract a JSON object from the LLM response."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        # Look for the first '{' and last '}' - handles LLMs that wrap in prose.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _parse_kind(value) -> StepKind | None:
        if not isinstance(value, str):
            return None
        try:
            return StepKind(value.strip().lower())
        except ValueError:
            return None