"""
Eval harness for summarizer_agent - two deterministic checks, not LLM-as-judge:

1. no_hallucinated_sources - calls the REAL summarizer_agent (makes a live
   Vertex AI call via llm_report, per agent_graph.py). Extracts anything
   filename-shaped from the generated report and checks it's a subset of the
   filenames actually given as input - summarizer_agent's prompt only ever
   passes candidate filenames + evaluation dicts, never raw resume content
   or real names, so any unrecognized filename-shaped token means the LLM
   invented a candidate that was never in evaluated_candidates.

2. empty_candidates_fallback - summarizer_agent short-circuits before
   touching the LLM when evaluated_candidates is empty, so this case makes
   NO API call and is free/instant to run. Checks the exact fallback string.

Usage:
    python eval/eval_summarizer_checks.py
    python eval/eval_summarizer_checks.py --threshold 0.85
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_graph import summarizer_agent  # noqa: E402

# Matches tokens shaped like a resume filename, e.g. resume_foo.txt / cand_1.pdf
FILENAME_PATTERN = re.compile(r"\b[\w\-]+\.(?:txt|pdf)\b", re.IGNORECASE)


def build_state(case: dict) -> dict:
    return {
        "job_description": case["job_description"],
        "query": "",
        "retrieved_candidates": [],
        "evaluated_candidates": case["evaluated_candidates"],
        "final_report": "",
        "delivery_status": None,
    }


def run_hallucination_case(case: dict) -> dict:
    result = {
        "id": case["id"],
        "passed": None,
        "problems": [],
        "report_excerpt": "",
        "error": None,
    }

    try:
        result_state = summarizer_agent(build_state(case))
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["passed"] = False
        return result

    report = result_state["final_report"]
    result["report_excerpt"] = report[:200] + ("..." if len(report) > 200 else "")

    expected_sources = {c["source"] for c in case["evaluated_candidates"]}
    mentioned = set(FILENAME_PATTERN.findall(report))

    unexpected = mentioned - expected_sources
    if unexpected:
        result["problems"].append(
            f"report references filename(s) not in input: {unexpected}"
        )

    result["passed"] = len(result["problems"]) == 0
    return result


def run_empty_fallback_case(case: dict) -> dict:
    result = {
        "id": case["id"],
        "passed": None,
        "problems": [],
        "report_excerpt": "",
        "error": None,
    }

    try:
        result_state = summarizer_agent(build_state(case))
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["passed"] = False
        return result

    report = result_state["final_report"]
    result["report_excerpt"] = report

    if report != case["expected_final_report"]:
        result["problems"].append(f"expected exact fallback string, got: {report!r}")

    result["passed"] = len(result["problems"]) == 0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        default=str(Path(__file__).parent / "fixtures" / "summarizer_checks.json"),
    )
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    cases = {c["id"]: c for c in json.loads(Path(args.fixtures).read_text())}

    results = [
        run_hallucination_case(cases["no_hallucinated_sources"]),
        run_empty_fallback_case(cases["empty_candidates_fallback"]),
    ]

    print("=" * 70)
    print("SUMMARIZER CHECKS EVAL")
    print("=" * 70)

    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"\n[{status}] {r['id']}")
        print(f"    report: {r['report_excerpt']!r}")
        for p in r["problems"]:
            print(f"    - {p}")
        if r["error"]:
            print(f"    ERROR: {r['error']}")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    score = passed / total if total else 1.0

    print("\n" + "=" * 70)
    print(
        f"RESULT: {passed}/{total} passed  (score={score:.2f}, threshold={args.threshold})"
    )
    print("=" * 70)

    sys.exit(0 if score >= args.threshold else 1)


if __name__ == "__main__":
    main()
