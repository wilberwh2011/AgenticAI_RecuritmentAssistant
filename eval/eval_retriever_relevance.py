"""
Eval harness for retriever relevance (rag_engine.search_candidates).

Builds a small, isolated vector store from synthetic fixture resumes in
eval/fixtures/retriever_resumes/, persisted to a TEMP directory (never the
production ./chroma_db - see the persist_directory parameter added to
build_vector_store for this exact reason). Runs each query fixture and
checks:
  - the expected top-ranked candidate is actually top-ranked
  - candidates that clearly shouldn't match don't appear in a small top-k
  - dedup-by-filename actually does something (the GCP architect resume is
    deliberately long enough to be split into multiple chunks; we confirm
    the raw similarity search returns duplicate chunks for that source, and
    that search_candidates collapses them to one)

This makes real Vertex AI embedding calls (both to build the store and to
run each query) - costlier and slower than the delivery/evaluator evals, so
consider running this less often (e.g. before a demo, or nightly in CI)
rather than on every PR.

Usage:
    python eval/eval_retriever_relevance.py
    python eval/eval_retriever_relevance.py --threshold 0.85
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_engine import build_vector_store, search_candidates  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESUMES_DIR = FIXTURES_DIR / "retriever_resumes"
QUERIES_FILE = FIXTURES_DIR / "retriever_queries.json"


def run_query_case(case: dict, vectorstore) -> dict:
    result = {
        "id": case["id"],
        "mode": case["assertion_mode"],
        "passed": None,
        "problems": [],
        "top_sources": [],
    }

    docs = search_candidates(case["query"], vectorstore, k=case.get("k", 3))
    sources = [Path(d.metadata.get("source", "unknown")).name for d in docs]
    result["top_sources"] = sources

    if case.get("expect_top_source"):
        if not sources or sources[0] != case["expect_top_source"]:
            result["problems"].append(
                f"expected top source {case['expect_top_source']!r}, got {sources[0] if sources else None!r}"
            )

    for excluded in case.get("expect_excluded_sources", []):
        if excluded in sources:
            result["problems"].append(
                f"expected {excluded!r} to be excluded from top-{case.get('k', 3)}, but it appeared"
            )

    result["passed"] = len(result["problems"]) == 0
    return result


def run_dedup_check(vectorstore) -> dict:
    """Confirms dedup is doing real work, not just trivially passing because
    nothing was ever duplicated in the first place."""
    result = {"id": "dedup_by_filename", "passed": None, "problems": []}

    query = "GCP AI architect Python Kubernetes RAG LangChain experience"

    # Raw similarity search, bypassing search_candidates' dedup, with a large
    # k so we're likely to see multiple chunks from the long GCP resume.
    raw_docs = vectorstore.similarity_search(query, k=10)
    raw_gcp_chunks = [
        d
        for d in raw_docs
        if Path(d.metadata.get("source", "")).name == "resume_gcp_architect.txt"
    ]

    if len(raw_gcp_chunks) < 2:
        result["problems"].append(
            f"expected the long GCP resume to be split into 2+ chunks and surfaced raw, "
            f"only saw {len(raw_gcp_chunks)} - dedup check is inconclusive, fixture resume "
            f"may need to be longer to reliably force multiple chunks"
        )

    # Now go through the real dedup path
    deduped_docs = search_candidates(query, vectorstore, k=10)
    deduped_sources = [
        Path(d.metadata.get("source", "unknown")).name for d in deduped_docs
    ]
    gcp_count_deduped = deduped_sources.count("resume_gcp_architect.txt")

    if gcp_count_deduped != 1:
        result["problems"].append(
            f"expected exactly 1 entry for resume_gcp_architect.txt after dedup, got {gcp_count_deduped}"
        )

    result["passed"] = len(result["problems"]) == 0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    temp_dir = tempfile.mkdtemp(prefix="eval_chroma_")
    print(
        f"Building isolated temp vector store at {temp_dir} (production ./chroma_db is untouched)..."
    )

    try:
        vectorstore = build_vector_store(
            resume_folder=str(RESUMES_DIR),
            persist_directory=temp_dir,
            force_rebuild=True,
        )

        queries = json.loads(QUERIES_FILE.read_text())
        query_results = [run_query_case(c, vectorstore) for c in queries]
        dedup_result = run_dedup_check(vectorstore)

        print("\n" + "=" * 70)
        print("RETRIEVER RELEVANCE EVAL")
        print("=" * 70)

        for r in query_results:
            status = "✅ PASS" if r["passed"] else "❌ FAIL"
            print(f"\n[{status}] {r['id']}")
            print(f"    top sources: {r['top_sources']}")
            for p in r["problems"]:
                print(f"    - {p}")

        status = "✅ PASS" if dedup_result["passed"] else "❌ FAIL"
        print(f"\n[{status}] {dedup_result['id']}")
        for p in dedup_result["problems"]:
            print(f"    - {p}")

        all_results = query_results + [dedup_result]
        passed = sum(1 for r in all_results if r["passed"])
        total = len(all_results)
        score = passed / total if total else 1.0

        print("\n" + "=" * 70)
        print(
            f"RESULT: {passed}/{total} passed  (score={score:.2f}, threshold={args.threshold})"
        )
        print("=" * 70)

        sys.exit(0 if score >= args.threshold else 1)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up temp vector store at {temp_dir}")


if __name__ == "__main__":
    main()
