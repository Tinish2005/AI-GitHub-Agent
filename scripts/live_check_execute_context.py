"""LIVE retrieval check for Execute's generate step (real Gemini embeddings).

Run from the project root, in PowerShell:

    python scripts\\live_check_execute_context.py

The Gemini key is read from the GEMINI_API_KEY environment variable, or from
your .env file if the variable is not set. The key is never printed.

It asks a few questions about the psf/requests repository and checks whether
the right file and function appear in the top 5 results. These are the same
results the Execute generate step receives as its code context.

Pass criteria (both must hold):
    - the expected FILE is in the top 5 for at least 4 of 5 questions
    - the expected FUNCTION is in the top 5 for at least 3 of 5 questions

Exit code: 0 = pass, 1 = fail, 2 = setup problem (no key, empty store).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOP_K = 5
MIN_FILE_HITS = 4
MIN_FUNCTION_HITS = 3

# Each case: question, file fragments that count as the right file,
# and name fragments that count as the right function.
CASES = [
    {
        "question": "How does HTTPAdapter.send work?",
        "files": ["adapters.py"],
        "names": ["HTTPAdapter.send"],
    },
    {
        "question": "Where are cookies merged into a prepared request?",
        "files": ["cookies.py", "sessions.py"],
        "names": ["merge_cookies", "prepare_request"],
    },
    {
        "question": "How does Session.request create and send a request?",
        "files": ["sessions.py"],
        "names": ["Session.request"],
    },
    {
        "question": "Where is redirect handling implemented?",
        "files": ["sessions.py"],
        "names": ["resolve_redirects", "rebuild_auth", "should_strip_auth"],
    },
    {
        "question": "How are query string parameters encoded into the URL?",
        "files": ["models.py", "utils.py"],
        "names": ["_encode_params", "prepare_url", "prepare_params"],
    },
]

# Information only: a question the repo cannot answer. A vector search always
# returns neighbours, so this cannot pass or fail. Look at the distances.
NO_EVIDENCE_QUESTION = "How does the Kubernetes scheduler choose a node for a pod?"


def _norm(path: str) -> str:
    return str(path).replace("\\", "/").lower()


def evaluate(store, cases=CASES, top_k: int = TOP_K) -> dict:
    """Run every case against the store. No printing, no network of its own."""
    rows = []
    for case in cases:
        results = store.query(case["question"], top_k=top_k)
        hits = []
        file_hit = False
        function_hit = False
        for r in results:
            meta = r.metadata or {}
            fpath = _norm(meta.get("file_path", ""))
            qname = str(meta.get("qualified_name", ""))
            file_ok = any(f.lower() in fpath for f in case["files"])
            name_ok = any(n.lower() in qname.lower() for n in case["names"])
            file_hit = file_hit or file_ok
            function_hit = function_hit or (file_ok and name_ok)
            hits.append({
                "qualified_name": qname,
                "file_path": meta.get("file_path", ""),
                "distance": r.distance,
                "relevant": file_ok,
            })
        rows.append({
            "question": case["question"],
            "expected_files": case["files"],
            "expected_names": case["names"],
            "hits": hits,
            "file_hit": file_hit,
            "function_hit": function_hit,
        })
    file_hits = sum(1 for r in rows if r["file_hit"])
    function_hits = sum(1 for r in rows if r["function_hit"])
    return {
        "rows": rows,
        "file_hits": file_hits,
        "function_hits": function_hits,
        "total": len(rows),
        "passed": file_hits >= MIN_FILE_HITS and function_hits >= MIN_FUNCTION_HITS,
    }


def _build_store():
    from backend.config import get_settings
    from backend.indexing.embeddings import EmbeddingService
    from backend.indexing.vector_store import VectorStore
    from backend.rag.gemini_embeddings import GeminiEmbeddingBackend

    settings = get_settings()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        key = settings.gemini_api_key.get_secret_value().strip()
    if not key:
        print("No Gemini key found. Set GEMINI_API_KEY or put it in your .env file.")
        return None
    model = (settings.gemini_embedding_model or "").strip() or None
    svc = EmbeddingService(backend=GeminiEmbeddingBackend(api_key=key, model=model))
    return VectorStore(persist_directory=settings.vector_db_path, embedding_service=svc)


def _indexed_files(store) -> list:
    metas = getattr(store, "_metadatas", None) or []
    return sorted({_norm(m.get("file_path", "")) for m in metas if m.get("file_path")})


def main() -> int:
    store = _build_store()
    if store is None:
        return 2
    count = store.count()
    files = _indexed_files(store)
    print(f"Store: {count} chunks from {len(files)} files")
    if count == 0:
        print("The store is empty. Index psf/requests first, then run this again.")
        return 2
    for case in CASES:
        if not any(any(f.lower() in p for p in files) for f in case["files"]):
            print(f"WARNING: no indexed file matches {case['files']} "
                  f"(not indexed? then a miss below is an indexing gap, not a search problem)")

    report = evaluate(store)
    for i, row in enumerate(report["rows"], start=1):
        label = "PASS" if row["function_hit"] else ("FILE-ONLY" if row["file_hit"] else "MISS")
        print(f"\n[{i}] {label}  {row['question']}")
        print(f"    expected file {row['expected_files']}  function {row['expected_names']}")
        if not row["hits"]:
            print("    no results returned (empty store or vector size mismatch)")
        for h in row["hits"]:
            mark = "ok " if h["relevant"] else "irrelevant"
            print(f"    - {mark:10} {h['qualified_name']}  ({h['file_path']})  d={h['distance']:.3f}")
        if not row["file_hit"]:
            print("    MISSING: the expected file is not in the top results")

    print(f"\nFile hits:     {report['file_hits']}/{report['total']} (need {MIN_FILE_HITS})")
    print(f"Function hits: {report['function_hits']}/{report['total']} (need {MIN_FUNCTION_HITS})")

    info = store.query(NO_EVIDENCE_QUESTION, top_k=TOP_K)
    if info:
        print(f"\nINFO (no pass/fail): '{NO_EVIDENCE_QUESTION}'")
        print(f"    closest distance {info[0].distance:.3f}; compare with the numbers above.")
        print("    Real answers should have clearly smaller distances than this one.")

    print("\nRESULT:", "PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
