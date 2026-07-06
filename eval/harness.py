"""Regression harness — run the pipeline over a fixed case set and check results
against expectation bounds, so quality drift (from prompt/rubric/model changes)
is caught instead of silently shipping.

Usage:
    python -m eval.harness                 # run every case in eval/cases.yaml
    python -m eval.harness --id vampire_hunter   # run one case (cheap smoke)
    python -m eval.harness --output runs/reg.json # also write JSON results

Because each case runs the full multi-agent loop, this makes real API calls —
it is a manual/CI-gated evaluation tool, not part of the unit test suite. The
pure scoring logic (``evaluate_case`` / ``load_cases``) is unit-tested offline.

If LangSmith env vars are set, every run is traced under a dedicated project so
you can inspect per-agent tokens, latency, and cost alongside these results.
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
from pathlib import Path

import yaml

# ── Ensure project root is importable, then load env ──────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from main import run_pipeline, SCORE_THRESHOLD  # noqa: E402
from utils.tracing import configure_tracing      # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CASES = Path(__file__).parent / "cases.yaml"
DEFAULT_PROJECT = "script-doctor-regression"


# ── Pure logic (unit-tested, no API) ──────────────────────────────────────────

def load_cases(path: str | Path = DEFAULT_CASES) -> list[dict]:
    """Load and lightly validate the regression cases file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError("Cases file must be a YAML list of case objects.")
    for i, case in enumerate(data):
        if not isinstance(case, dict) or "id" not in case or "prompt" not in case:
            raise ValueError(f"Case #{i} must be a mapping with 'id' and 'prompt' keys.")
    return data


def evaluate_case(expect: dict, result: dict, threshold: int = SCORE_THRESHOLD) -> list[str]:
    """Check one run's result against expectation bounds.

    Returns a list of human-readable failure strings; empty means the case passed.
    """
    if "error" in result:
        return [f"run error: {result['error']}"]

    failures: list[str] = []
    score = result.get("final_score")
    iters = result.get("iterations")

    if score is None:
        return ["result has no final_score"]

    if "min_final_score" in expect and score < expect["min_final_score"]:
        failures.append(f"final_score {score} < min {expect['min_final_score']}")
    if "max_final_score" in expect and score > expect["max_final_score"]:
        failures.append(f"final_score {score} > max {expect['max_final_score']}")

    if iters is not None:
        if "min_iterations" in expect and iters < expect["min_iterations"]:
            failures.append(f"iterations {iters} < min {expect['min_iterations']}")
        if "max_iterations" in expect and iters > expect["max_iterations"]:
            failures.append(f"iterations {iters} > max {expect['max_iterations']}")

    if expect.get("require_converged") and score < threshold:
        failures.append(f"did not converge (score {score} < threshold {threshold})")

    return failures


# ── Execution (makes API calls) ───────────────────────────────────────────────

def _run_one(case: dict) -> dict:
    """Execute the pipeline for a single case and collect metrics."""
    start = time.time()
    final_state, _history, _ts = run_pipeline(case["prompt"], save_files=False)
    elapsed = round(time.time() - start, 1)
    return {
        "id": case["id"],
        "final_score": final_state.get("score"),
        "iterations": final_state.get("iteration_count"),
        "scores_progression": final_state.get("scores_progression", []),
        "last_scores": final_state.get("last_scores", {}),
        "elapsed_s": elapsed,
    }


def run_regression(cases: list[dict], threshold: int = SCORE_THRESHOLD) -> tuple[list[dict], bool]:
    """Run every case, evaluate it, and return (results, all_passed)."""
    results: list[dict] = []
    all_passed = True
    for case in cases:
        logger.info("─" * 60)
        logger.info("Running case: %s", case["id"])
        try:
            res = _run_one(case)
        except Exception as exc:  # noqa: BLE001 — record any failure, keep going
            logger.error("Case %s crashed: %s", case["id"], exc)
            res = {"id": case["id"], "error": str(exc)}

        res["failures"] = evaluate_case(case.get("expect", {}), res, threshold)
        res["passed"] = not res["failures"]
        all_passed = all_passed and res["passed"]
        results.append(res)
    return results, all_passed


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(results: list[dict], threshold: int) -> None:
    sep = "=" * 72
    print(f"\n{sep}\n  SCRIPT DOCTOR — REGRESSION REPORT  (threshold ≥ {threshold})\n{sep}")
    print(f"  {'CASE':<20} {'RESULT':<7} {'SCORE':<6} {'ITERS':<6} {'PROGRESSION':<16} {'TIME':<7}")
    print(f"  {'-'*20} {'-'*7} {'-'*6} {'-'*6} {'-'*16} {'-'*7}")
    for r in results:
        verdict = "PASS" if r.get("passed") else "FAIL"
        score = r.get("final_score", "—")
        iters = r.get("iterations", "—")
        prog = "→".join(str(s) for s in r.get("scores_progression", [])) or "—"
        elapsed = f"{r['elapsed_s']}s" if "elapsed_s" in r else "—"
        print(f"  {r['id']:<20} {verdict:<7} {str(score):<6} {str(iters):<6} {prog:<16} {elapsed:<7}")
        for f in r.get("failures", []):
            print(f"      ↳ {f}")
    passed = sum(1 for r in results if r.get("passed"))
    print(f"{sep}\n  {passed}/{len(results)} cases passed\n{sep}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-22s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "chromadb", "sentence_transformers", "urllib3", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run Script Doctor regression cases.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="Path to cases YAML.")
    parser.add_argument("--id", nargs="*", help="Only run cases with these ids.")
    parser.add_argument("--output", help="Write JSON results to this path.")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="LangSmith project name.")
    args = parser.parse_args(argv)

    configure_tracing(project=args.project)

    cases = load_cases(args.cases)
    if args.id:
        wanted = set(args.id)
        cases = [c for c in cases if c["id"] in wanted]
        if not cases:
            logger.error("No cases matched ids: %s", args.id)
            return 2

    logger.info("Running %d regression case(s)...", len(cases))
    results, all_passed = run_regression(cases)
    print_report(results, SCORE_THRESHOLD)

    if args.output:
        payload = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "threshold": SCORE_THRESHOLD,
            "all_passed": all_passed,
            "results": results,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Results written to %s", args.output)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
