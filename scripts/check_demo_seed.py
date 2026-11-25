"""Check the demo seed file before you commit and deploy it.

Run from the project root, in PowerShell:

    python scripts\\check_demo_seed.py

It checks that the file exists, is not damaged, is a sensible size, and is not
blocked by your .gitignore. The .gitignore check is a best guess. Inside your
real git folder you can also run:  git check-ignore -v demo_seed/psf_requests.json.gz
(no output means "not ignored"). Finally, look at the file on GitHub: if it is
not there, Render will not have it and the demo will be empty.

Exit code: 0 = all good, 1 = a problem was found.
"""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WARN_MB = 50
MAX_MB = 90  # GitHub refuses files over 100 MB


def is_ignored(rel_path: str, gitignore_text: str) -> bool:
    """Best-effort .gitignore check: does any rule ignore this file?"""
    parts = [p for p in rel_path.replace("\\", "/").strip("/").split("/") if p]
    ignored = False
    for raw in gitignore_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:]
        dir_only = line.endswith("/")
        pattern = line.rstrip("/")
        anchored = pattern.startswith("/") or "/" in pattern
        pattern = pattern.lstrip("/")
        matched = False
        if anchored:
            for i in range(1, len(parts) + 1):
                if dir_only and i == len(parts):
                    continue
                if fnmatch.fnmatchcase("/".join(parts[:i]), pattern):
                    matched = True
                    break
        else:
            for i, part in enumerate(parts):
                if dir_only and i == len(parts) - 1:
                    continue
                if fnmatch.fnmatchcase(part, pattern):
                    matched = True
                    break
        if matched:
            ignored = not negate
    return ignored


def check(seed_path: Path, project_root: Path, configured_model: str = "") -> list:
    """Return a list of (level, message); level is OK, WARN or FAIL."""
    from backend.indexing.demo_seed import DemoSeedError, read_seed

    results = []
    if not seed_path.is_file():
        return [("FAIL", f"Seed file not found: {seed_path}. Run scripts\\build_demo_seed.py first.")]
    size_mb = seed_path.stat().st_size / 1e6
    results.append(("OK", f"Seed file found: {seed_path} ({size_mb:.1f} MB)"))
    if size_mb > MAX_MB:
        results.append(("FAIL", f"The file is {size_mb:.0f} MB. GitHub refuses files over 100 MB."))
    elif size_mb > WARN_MB:
        results.append(("WARN", f"The file is {size_mb:.0f} MB. Every rebuild adds this much to your git history."))

    try:
        info, _ = read_seed(seed_path)
    except DemoSeedError as exc:
        results.append(("FAIL", f"The seed file is not usable: {exc}"))
        return results
    manifest = info["manifest"]
    results.append((
        "OK",
        f"Seed is valid: {manifest['repo']}, commit {str(manifest.get('commit', ''))[:7] or 'unknown'}, "
        f"{info['chunks']} chunks, {info['dimension']} numbers each, model {info['model'] or 'not recorded'}",
    ))

    wanted = (configured_model or "").strip()
    if wanted and info["model"]:
        norm = lambda m: m if m.startswith("models/") else f"models/{m}"  # noqa: E731
        if norm(wanted) != norm(info["model"]):
            results.append(("WARN", f"GEMINI_EMBEDDING_MODEL is {wanted} but the seed was built with {info['model']}."))
    if not wanted:
        results.append(("WARN", f"GEMINI_EMBEDDING_MODEL is not set. Set it to {info['model']} here and on Render."))

    try:
        rel = seed_path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        results.append(("WARN", "The seed file is outside the project folder, so it will not be deployed."))
        return results
    gitignore = project_root / ".gitignore"
    if gitignore.is_file() and is_ignored(rel, gitignore.read_text(encoding="utf-8", errors="replace")):
        results.append(("FAIL", f"Your .gitignore blocks {rel}. Remove or change that rule, or Render will not get the demo."))
    else:
        results.append(("OK", f"Your .gitignore does not block {rel} (best guess)."))
    return results


def main(argv: list) -> int:
    from backend.config import PROJECT_ROOT, get_settings

    settings = get_settings()
    results = check(settings.demo_seed_path, PROJECT_ROOT, settings.gemini_embedding_model)
    for level, message in results:
        print(f"[{level:4}] {message}")
    failed = any(level == "FAIL" for level, _ in results)
    print("\nRESULT:", "FAIL" if failed else "PASS")
    if not failed:
        print("Last step you must do yourself: confirm the file is visible on GitHub after you push.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
