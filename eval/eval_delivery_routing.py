"""
Eval harness for delivery_agent's routing decision (decide_delivery).

This calls the REAL decide_delivery function, which makes a live Vertex AI
call per fixture (temperature 0.2, per agent_graph.py's shared `llm`). That
means:
  - Running this costs a small amount of Vertex AI usage per run.
  - Results for "strict" cases should be stable given temperature 0.2 + clear
    inputs, but are not mathematically guaranteed reproducible run to run.
  - "observe" cases are intentionally ambiguous inputs where we don't assert
    a specific outcome — we just log what happened so a human can sanity
    check it, rather than have an inherently fuzzy case fail CI.

Usage:
    python eval/eval_delivery_routing.py
    python eval/eval_delivery_routing.py --threshold 0.85
    python eval/eval_delivery_routing.py --fixtures eval/fixtures/delivery_routing.json
"""

import argparse
import json
import sys
from pathlib import Path

# Make the project root importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_graph import decide_delivery  # noqa: E402

DUMMY_REPORT = "CANDIDATE 1: resume_test.pdf\nScore: 8/10\nStrong match.\n"


def run_case(case: dict) -> dict:
    """Runs decide_delivery for one fixture and checks it against expectations
    (only for strict cases). Returns a result dict for reporting."""
    result = {
        "id": case["id"],
        "description": case["description"],
        "mode": case["assertion_mode"],
        "input": case["input"],
        "passed": None,  # None = not scored (observe mode)
        "actual_tool": None,
        "actual_recipient": None,
        "actual_fallback_reason": None,
        "error": None,
    }

    try:
        decision = decide_delivery(case["input"], DUMMY_REPORT)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["passed"] = False if case["assertion_mode"] == "strict" else None
        return result

    result["actual_tool"] = decision.get("tool")
    result["actual_recipient"] = decision.get("args", {}).get("recipient")
    result["actual_fallback_reason"] = decision.get("fallback_reason")

    if case["assertion_mode"] == "observe":
        # Nothing to assert — just report what happened
        return result

    checks = [decision.get("tool") == case["expected_tool"]]
    if case.get("expected_recipient") is not None:
        checks.append(
            decision.get("args", {}).get("recipient") == case["expected_recipient"]
        )
    if "expected_fallback_reason" in case:
        expected_fb = case["expected_fallback_reason"]
        actual_fb = decision.get("fallback_reason")
        if isinstance(expected_fb, list):
            checks.append(actual_fb in expected_fb)
        else:
            checks.append(actual_fb == expected_fb)

    result["passed"] = all(checks)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        default=str(Path(__file__).parent / "fixtures" / "delivery_routing.json"),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Minimum pass rate (strict cases only) to exit 0",
    )
    args = parser.parse_args()

    fixtures = json.loads(Path(args.fixtures).read_text())

    results = [run_case(c) for c in fixtures]

    strict_results = [r for r in results if r["mode"] == "strict"]
    observe_results = [r for r in results if r["mode"] == "observe"]

    print("=" * 70)
    print("DELIVERY ROUTING EVAL")
    print("=" * 70)

    for r in results:
        if r["mode"] == "strict":
            status = "✅ PASS" if r["passed"] else "❌ FAIL"
        else:
            status = "ℹ️  OBSERVE"
        print(f"\n[{status}] {r['id']}")
        print(f"    input: {r['input']!r}")
        print(
            f"    tool={r['actual_tool']} recipient={r['actual_recipient']} fallback={r['actual_fallback_reason']}"
        )
        if r["error"]:
            print(f"    ERROR: {r['error']}")

    passed = sum(1 for r in strict_results if r["passed"])
    total = len(strict_results)
    score = passed / total if total else 1.0

    print("\n" + "=" * 70)
    print(
        f"STRICT CASES: {passed}/{total} passed  (score={score:.2f}, threshold={args.threshold})"
    )
    print(
        f"OBSERVE CASES: {len(observe_results)} logged, not scored — review manually above"
    )
    print("=" * 70)

    sys.exit(0 if score >= args.threshold else 1)


if __name__ == "__main__":
    main()
