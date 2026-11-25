"""
Planners.

A Planner turns a user goal into a Plan. Two implementations ship in


    - RuleBasedPlanner: deterministic, no LLM. Picks a plan template
      based on words in the goal. Fast, cheap, testable. Every step it
      lists is one the Execute engine can really run, and the plan gets
      longer or shorter to fit the goal (2 to 5 steps).

    - LLMPlanner: wraps an LLMClient and asks it to produce a plan in
      a strict JSON shape. Falls back to the rule-based planner if
      the LLM returns malformed output ("graceful degradation").
"""

from __future__ import annotations

import json
import re
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
    "codebase. Given a user goal, determine the appropriate number of logical "
    "steps needed to accomplish it (typically 2 to 5 steps depending on complexity). "
    "Output a JSON object with a 'steps' array. Each step must have: 'kind' (one of: "
    "retrieve, analyze, github_read, generate, validate) and 'description'. "
    "Only these five kinds can be run. 'retrieve' searches the indexed code; "
    "'github_read' lists the repository's open issues; 'generate' proposes a code change "
    "as a diff (it is not applied); 'validate' only checks that the diff fits the real "
    "repository and passes syntax and import checks (it does not run tests or lint). "
    "Return ONLY the JSON - no prose."
)

# Words that decide which plan a goal gets. Common word forms are listed on purpose.
FIX_VERBS = frozenset({
    "fix", "fixes", "fixed", "fixing", "repair", "repairs", "repaired", "repairing",
    "resolve", "resolves", "resolved", "resolving", "patch", "patches", "patched", "patching",
    "solve", "solves", "solved", "solving", "debug", "debugs", "debugged", "debugging",
})
SYMPTOM_WORDS = frozenset({
    "bug", "bugs", "buggy", "broken", "error", "errors", "crash", "crashes", "crashed", "crashing",
    "fail", "fails", "failed", "failing", "failure", "failures", "regression", "regressions",
})
REVIEW_WORDS = frozenset({
    "review", "reviews", "reviewing", "audit", "audits", "auditing", "explain", "explains",
    "explaining", "understand", "understanding", "describe", "describes", "summarize",
    "summarise", "summary", "overview",
})
CHANGE_WORDS = frozenset({
    "refactor", "refactors", "refactoring", "refactored", "optimize", "optimise", "optimizing",
    "optimising", "improve", "improves", "improving", "improvement", "improvements",
    "implement", "implements", "implementing", "build", "building", "feature", "features",
    "workflow", "add", "adds", "adding", "create", "creates", "creating", "update", "updates",
    "updating", "change", "changes", "changing", "modify", "modifies", "modifying", "rename",
    "renaming", "remove", "removing", "replace", "replacing", "rewrite", "rewriting",
    "migrate", "migrating", "convert", "converting", "extend", "extending",
})
SUGGEST_WORDS = frozenset({
    "suggest", "suggests", "suggestion", "suggestions", "outline", "outlines", "draft",
    "propose", "proposes", "recommend", "recommends", "design", "plan", "sketch", "idea", "ideas",
})
GITHUB_WORDS = frozenset({"issue", "issues", "github"})

# What Execute really does for each kind of step. Keep these texts honest.
_RETRIEVE_FIX = "Search the indexed code for chunks related to the reported problem."
_RETRIEVE_CHANGE = "Search the indexed code for the parts this change affects."
_RETRIEVE_AREA = "Search the indexed code for the area the change should touch."
_RETRIEVE_ASK = "Search the indexed code for chunks related to the question."
_RETRIEVE_REVIEW = "Search the indexed code for what the goal refers to."
_GITHUB_READ = "List the repository's open issues for context (needs owner and repo in the request)."
_VALIDATE = (
    "Check that the diff fits the real repository files and passes syntax and import checks. "
    "Lint, tests and security scans are not run."
)


def _words(goal: str) -> list:
    """Lowercased words of the goal. Punctuation and capitals do not matter."""
    return re.findall(r"[a-z0-9]+", goal.lower())


def _keywords(goal: str) -> set:
    """Return the set of lowercased words for keyword matching."""
    return set(_words(goal))


def _mentions_pull_request(tokens: list) -> bool:
    """True for the word PR (or PRs) and for the phrase "pull request"."""
    if "pr" in tokens or "prs" in tokens:
        return True
    return any(
        a == "pull" and b in ("request", "requests")
        for a, b in zip(tokens, tokens[1:])
    )


class RuleBasedPlanner:
    """Deterministic planner that picks a template based on the goal."""

    strategy: str = "rule_based"

    def plan(self, goal: str) -> Plan:
        if not goal or not goal.strip():
            raise ValueError("Goal must not be empty.")

        tokens = _words(goal)
        template = self._pick_template(set(tokens), tokens)
        steps = tuple(
            PlanStep(id=i + 1, kind=kind, description=desc)
            for i, (kind, desc) in enumerate(template)
        )
        return Plan(goal=goal.strip(), steps=steps, strategy=self.strategy)

    @staticmethod
    def _pick_template(words: set, tokens: list | None = None) -> list:
        """Pick a plan that fits the goal. Plans have 2 to 5 steps.

        When a goal mixes intents, the strongest one wins:
        fix > pull request > change > symptom (unless asked to explain)
        > suggestion > review > GitHub > question.
        """
        tokens = tokens if tokens is not None else sorted(words)
        asks_github = bool(words & GITHUB_WORDS)
        review = bool(words & REVIEW_WORDS)

        if words & FIX_VERBS or (words & SYMPTOM_WORDS and not review):
            template = [
                (StepKind.RETRIEVE, _RETRIEVE_FIX),
                (StepKind.ANALYZE, "Work out the likely cause from the retrieved code."),
                (StepKind.GENERATE, "Ask the AI to propose a fix as a unified diff. The diff is not applied anywhere."),
                (StepKind.VALIDATE, _VALIDATE),
            ]
        elif _mentions_pull_request(tokens):
            template = [
                (StepKind.RETRIEVE, _RETRIEVE_AREA),
                (StepKind.ANALYZE, "Work out the scope and impact of the change."),
                (
                    StepKind.GENERATE,
                    "Ask the AI to propose the change as a unified diff for review. "
                    "Opening the pull request is a separate step you start after reviewing it.",
                ),
                (StepKind.VALIDATE, _VALIDATE),
            ]
        elif words & CHANGE_WORDS:
            template = [
                (StepKind.RETRIEVE, _RETRIEVE_CHANGE),
                (StepKind.ANALYZE, "Analyze how that code works and what the change must respect."),
                (StepKind.GENERATE, "Ask the AI to propose the change as a unified diff. The diff is not applied anywhere."),
                (StepKind.VALIDATE, _VALIDATE),
            ]
        elif words & SUGGEST_WORDS:
            template = [
                (StepKind.RETRIEVE, "Search the indexed code for the relevant parts."),
                (StepKind.ANALYZE, "Analyze how that code works and where an improvement fits."),
                (
                    StepKind.GENERATE,
                    "Ask the AI to write the suggested change as a unified diff. It is not validated.",
                ),
            ]
        elif review:
            template = [
                (StepKind.RETRIEVE, _RETRIEVE_REVIEW),
                (StepKind.ANALYZE, "Summarize that code and point out risks or unclear parts."),
            ]
        elif asks_github:
            return [
                (StepKind.GITHUB_READ, "List the repository's open issues (needs owner and repo in the request)."),
                (StepKind.ANALYZE, "Summarize what the issues say."),
            ]
        else:
            return [
                (StepKind.RETRIEVE, _RETRIEVE_ASK),
                (StepKind.ANALYZE, "Answer the question from the retrieved code."),
            ]

        if asks_github:
            # The goal mentions issues or GitHub: read the open issues right after the search.
            template.insert(1, (StepKind.GITHUB_READ, _GITHUB_READ))
        return template


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