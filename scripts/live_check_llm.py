"""LIVE check of the AI model: can it write a usable fix? (uses ONE AI call)

Run from the project root, in PowerShell:

    python scripts\\live_check_llm.py
    python scripts\\live_check_llm.py --model YOUR-MODEL-ID

Use --model to test a different Gemini model (copy the exact model id from Google
AI Studio) before you set GEMINI_LLM_MODEL on Render. The check gives the AI a tiny
bug, asks for a fix as a diff, and checks that the diff is valid and fits the file.
Your Gemini key comes from GEMINI_API_KEY or your .env file and is never printed.
Exit code: 0 = pass, 1 = fail, 2 = could not start.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"
GOAL = "Fix the add function so that it adds the two numbers instead of subtracting them."


def run_check(llm) -> list:
    """Ask the AI for a fix and check it. Returns [(PASS|FAIL, message)]."""
    from backend.agent.fix_generator import FixGenerator
    from backend.agent.validation_checks import SyntaxCheck
    from backend.agent.validation_pipeline import ValidationPipeline

    results: list = []
    started = time.time()
    proposal = FixGenerator(llm=llm).propose(GOAL, f"File calc.py:\n{SAMPLE}")
    results.append(("INFO", f"The AI answered in {time.time() - started:.1f} seconds."))
    results.append((
        "PASS" if proposal.is_valid else "FAIL",
        "The answer is a valid diff." if proposal.is_valid else f"The answer is not a valid diff: {proposal.validation_error}",
    ))
    if not proposal.is_valid:
        return results

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base"
        base.mkdir()
        (base / "calc.py").write_text(SAMPLE, encoding="utf-8")
        verdict = ValidationPipeline(checks=(SyntaxCheck(),), base_root=base, sandbox_parent=Path(tmp) / "box").validate(proposal)
    results.append((
        "PASS" if verdict.passed else "FAIL",
        "The diff fits the file and the result is valid Python." if verdict.passed else f"The diff does not fit the file: {verdict.error}",
    ))
    adds = [line for line in proposal.diff.splitlines() if line.startswith("+") and "a + b" in line]
    results.append((
        "PASS" if adds else "FAIL",
        "The fix really changes the code to add." if adds else "The diff does not contain the change to 'a + b'.",
    ))
    return results


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="Check that the AI model can write a usable fix.")
    parser.add_argument("--model", default=None, help="model id to test (default: GEMINI_LLM_MODEL)")
    args = parser.parse_args(argv[1:])

    from backend.config import get_settings
    from backend.rag.gemini_llm import GeminiLLMClient

    settings = get_settings()
    key = settings.gemini_api_key.get_secret_value()
    if not key:
        print("No Gemini key found. Put GEMINI_API_KEY in your .env file.")
        return 2
    model = args.model or settings.gemini_llm_model
    print(f"Testing model: {model} (this uses one AI call)")
    try:
        results = run_check(GeminiLLMClient(api_key=key, model=model))
    except Exception as exc:
        print(f"The AI call failed: {type(exc).__name__}: {str(exc)[:200]}")
        print("A 404 usually means the model id is wrong or not available to this SDK. A 429 means a quota limit.")
        return 2
    for level, message in results:
        print(f"[{level:4}] {message}")
    failed = any(level == "FAIL" for level, _ in results)
    print("\nRESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
