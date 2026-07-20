"""
Eval harness for the evaluator agent's scoring function (evaluate_candidates_batch).

Two kinds of checks here, deliberately separated:

1. Fixture-driven checks (this file's `run_fixture_cases`) - call the REAL
   evaluate_candidates_batch, which makes a live Vertex AI call per case
   (llm_eval, temperature=0, per agent_graph.py). Costs a small amount of
   Vertex AI usage per run. Every case is checked for schema validity and
   count integrity regardless of assertion_mode; cases with stronger_source/
   weaker_source set are additionally checked for correct relative ordering.
   "observe" cases skip the ordering assertion and just log scores.

2. Mocked check (`test_parse_failure_fallback`) - no API call at all. We
   swap in a fake llm_eval whose .invoke() always returns malformed JSON,
   to confirm evaluate_candidates_batch degrades to its documented fallback
   (score 0, "Parse error") instead of crashing. This is intentionally NOT
   fixture-driven since it's testing an error path, not model judgment.

Usage:
    python eval/eval_evaluator_scoring.py
    python eval/eval_evaluator_scoring.py --threshold 0.85
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_graph import llm_eval  # noqa: E402
from tools import evaluate_candidates_batch  # noqa: E402

REQUIRED_KEYS = {"source", "content", "evaluation", "score"}
REQUIRED_EVAL_KEYS = {"score", "meets_requirements", "strengths", "gaps", "summary"}


def check_schema(evaluated: list) -> list:
    """Returns a list of problem strings; empty list means schema is clean."""
    problems = []
    for i, item in enumerate(evaluated):
        missing = REQUIRED_KEYS - item.keys()
        if missing:
            problems.append(f"item {i}: missing top-level keys {missing}")
            continue
        if not isinstance(item["score"], (int, float)):
            problems.append(f"item {i}: score is not numeric ({type(item['score'])})")
        ev = item.get("evaluation")
        if not isinstance(ev, dict):
            problems.append(f"item {i}: evaluation is not a dict ({type(ev)})")
            continue
        missing_ev = REQUIRED_EVAL_KEYS - ev.keys()
        if missing_ev:
            problems.append(f"item {i}: evaluation missing keys {missing_ev}")
    return problems


def run_fixture_case(case: dict) -> dict:
    result = {
        "id": case["id"],
        "mode": case["assertion_mode"],
        "passed": None,
        "problems": [],
        "scores": {},
        "error": None,
    }

    try:
        evaluated = evaluate_candidates_batch(
            case["candidates"], case["job_description"], llm_eval
        )
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["passed"] = False
        return result

    result["scores"] = {item["source"]: item["score"] for item in evaluated}

    # Count integrity - checked on every case regardless of mode
    if len(evaluated) != len(case["candidates"]):
        result["problems"].append(
            f"count mismatch: {len(evaluated)} results for {len(case['candidates'])} candidates"
        )

    # Schema validity - checked on every case regardless of mode
    result["problems"].extend(check_schema(evaluated))

    if case["assertion_mode"] == "observe":
        result["passed"] = None if not result["problems"] else False
        return result

    # Ordering check, only when the fixture specifies it
    if "stronger_source" in case and "weaker_source" in case:
        stronger_score = result["scores"].get(case["stronger_source"])
        weaker_score = result["scores"].get(case["weaker_source"])
        if stronger_score is None or weaker_score is None:
            result["problems"].append(
                "could not find expected sources in results to compare"
            )
        elif not (stronger_score > weaker_score):
            result["problems"].append(
                f"expected {case['stronger_source']} ({stronger_score}) > "
                f"{case['weaker_source']} ({weaker_score}), but it wasn't"
            )

    result["passed"] = len(result["problems"]) == 0
    return result


def test_parse_failure_fallback() -> dict:
    """No API call - mocks llm_eval to return malformed JSON and checks the
    documented fallback behavior (score 0, 'Parse error', correct count)."""

    class MockLLMEval:
        def invoke(self, messages):
            return SimpleNamespace(content="this is not valid json at all")

    candidates = [
        {"source": "a.txt", "content": "irrelevant for this test", "github_data": None},
        {"source": "b.txt", "content": "irrelevant for this test", "github_data": None},
    ]

    result = {
        "id": "parse_failure_fallback",
        "mode": "strict",
        "passed": None,
        "problems": [],
        "error": None,
    }
    try:
        evaluated = evaluate_candidates_batch(
            candidates, "irrelevant JD", MockLLMEval()
        )
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["passed"] = False
        return result

    if len(evaluated) != len(candidates):
        result["problems"].append(
            f"expected {len(candidates)} results, got {len(evaluated)}"
        )
    for item in evaluated:
        if item.get("score") != 0:
            result["problems"].append(
                f"expected fallback score 0, got {item.get('score')}"
            )
        if item.get("evaluation", {}).get("summary") != "Parse error":
            result["problems"].append(
                f"expected fallback summary 'Parse error', got {item.get('evaluation')}"
            )

    result["passed"] = len(result["problems"]) == 0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        default=str(Path(__file__).parent / "fixtures" / "evaluator_scoring.json"),
    )
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    fixtures = json.loads(Path(args.fixtures).read_text())
    fixture_results = [run_fixture_case(c) for c in fixtures]
    mock_result = test_parse_failure_fallback()

    print("=" * 70)
    print("EVALUATOR SCORING EVAL")
    print("=" * 70)

    for r in fixture_results:
        if r["mode"] == "observe":
            status = (
                "ℹ️  OBSERVE" if not r["problems"] else "⚠️  OBSERVE (problems found)"
            )
        else:
            status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"\n[{status}] {r['id']}")
        print(f"    scores: {r['scores']}")
        if r["problems"]:
            for p in r["problems"]:
                print(f"    - {p}")
        if r["error"]:
            print(f"    ERROR: {r['error']}")

    print(
        f"\n[{'✅ PASS' if mock_result['passed'] else '❌ FAIL'}] {mock_result['id']} (mocked, no API call)"
    )
    if mock_result["problems"]:
        for p in mock_result["problems"]:
            print(f"    - {p}")
    if mock_result["error"]:
        print(f"    ERROR: {mock_result['error']}")

    scored = [r for r in fixture_results if r["mode"] != "observe"] + [mock_result]
    passed = sum(1 for r in scored if r["passed"])
    total = len(scored)
    score = passed / total if total else 1.0

    print("\n" + "=" * 70)
    print(
        f"SCORED CASES: {passed}/{total} passed  (score={score:.2f}, threshold={args.threshold})"
    )
    observe_count = sum(1 for r in fixture_results if r["mode"] == "observe")
    print(f"OBSERVE CASES: {observe_count} logged, not scored — review manually above")
    print("=" * 70)

    sys.exit(0 if score >= args.threshold else 1)


if __name__ == "__main__":
    main()
