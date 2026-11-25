"""Restore the demo seed into a store folder (what the server does at startup).

Run from the project root, in PowerShell:

    python scripts\\restore_demo_seed.py --target C:\\temp\\demo_test

Without --target it uses your real vector_db folder. A store that already has
data is never changed, so this is safe to run. To test the restore, point
--target at a new empty folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description="Restore the demo seed.")
    parser.add_argument("--target", default=None, help="store folder (default: vector_db_path)")
    parser.add_argument("--seed", default=None, help="seed file (default: demo_seed_path)")
    args = parser.parse_args(argv[1:])

    from backend.config import get_settings
    from backend.indexing.demo_seed import restore_demo_seed

    settings = get_settings()
    seed = Path(args.seed) if args.seed else settings.demo_seed_path
    target = Path(args.target) if args.target else settings.vector_db_path
    status = restore_demo_seed(
        seed, target, configured_model=settings.gemini_embedding_model, enabled=True,
    )
    print(f"seed:    {seed}")
    print(f"target:  {target}")
    print(f"result:  {status.message}")
    if status.warning:
        print(f"WARNING: {status.warning}")
    if status.restored:
        print(f"         {status.chunks} chunks, {status.dimension} numbers each, model {status.model}")
    return 0 if status.restored else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
