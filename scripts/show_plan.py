"""Print the plan the rule-based planner makes for a goal (no network, no AI).

Run from the project root, in PowerShell:

    python scripts\\show_plan.py "refactor request handling"
    python scripts\\show_plan.py "fix the login bug" "explain how sessions work"

With no goal it prints plans for a set of sample goals.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLES = [
    "refactor request handling",
    "fix the login bug",
    "login bugs keep crashing the app",
    "add exponential backoff to retries",
    "open a pull request for the retry change",
    "suggest how to simplify redirects",
    "explain how sessions work",
    "list open issues",
    "fix the bug in issue 42",
    "how does the HTTPAdapter send requests?",
]


def render(plan) -> str:
    lines = [f"Goal: {plan.goal}   ({plan.step_count} steps)"]
    for step in plan.steps:
        lines.append(f"  {step.id}. [{step.kind.value}] {step.description}")
    return "\n".join(lines)


def main(argv: list) -> int:
    from backend.agent.planner import RuleBasedPlanner

    goals = argv[1:] or SAMPLES
    planner = RuleBasedPlanner()
    for goal in goals:
        print(render(planner.plan(goal)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
