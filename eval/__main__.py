"""Eval entry point.

  python -m eval metrics [--runs 3]    # repeated full pipeline runs, scored
  python -m eval metrics --clean       # negative control: bug-free SUT
  python -m eval injection             # poisoned-artifact resistance probes

All need GEMINI_API_KEY; `metrics` additionally needs Docker (SUT container
is started fresh per run so flaky counters stay reproducible).
Results are written to .qe_runs/eval-<timestamp>/.
"""

import argparse
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from eval.harness import (
    dump_defects,
    render_clean_results,
    render_results,
    run_pipeline_once,
    score_clean_run,
    score_run,
)
from eval.injection import (
    VARIANTS,
    render_injection_results,
    run_response_variant,
    run_variant,
)
from eval.replay import replay_defect
from qe_agent import config, sandbox
from qe_agent.graph import build_graph
from qe_agent.report import operation_coverage, scenario_funnel


def _out_dir() -> Path:
    out = config.RUNS_DIR / f"eval-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def cmd_metrics(args: argparse.Namespace) -> None:
    out = _out_dir()
    port = urlparse(config.sut_base_url()).port or 8000
    sandbox.ensure_infra()
    scores = []
    coverage_lines = []
    for i in range(1, args.runs + 1):
        mode = "clean (negative control)" if args.clean else "planted-bug"
        print(f"[eval] run {i}/{args.runs}: starting fresh {mode} SUT container...")
        # clean mode also disables flakiness: the control must be bug-free
        sandbox.start_sut_container(
            host_port=port,
            clean=args.clean,
            flaky_every="0" if args.clean else None,
        )
        try:
            defects, result = run_pipeline_once(build_graph())
            # replay unmatched claims while this run's SUT is still up
            with httpx.Client(base_url=config.sut_base_url(), timeout=10.0) as client:

                def replay(defect):
                    return replay_defect(defect, client)

                score = (
                    score_clean_run(defects, replay) if args.clean else score_run(defects, replay)
                )
        finally:
            sandbox.stop_sut_container()
        (out / f"run-{i}-defects.json").write_text(dump_defects(defects))
        scores.append(score)
        covered, uncovered = operation_coverage(result)
        funnel = scenario_funnel(result)
        executed_n = sum(1 for row in funnel if row["status"] == "executed")
        coverage_lines.append(
            f"- run {i}: operations {len(covered)}/{len(covered) + len(uncovered)}"
            + (f" (uncovered: {', '.join(uncovered)})" if uncovered else "")
            + f" | scenarios executed {executed_n}/{len(funnel)}"
        )
        print(f"[eval] {coverage_lines[-1][2:]}")
        if args.clean:
            print(
                f"[eval] clean run {i}: {len(defects)} defects, "
                f"{len(score.false_alarms)} confirmed false alarms, "
                f"{len(score.reproduced)} reproduced spec gaps"
            )
        else:
            print(
                f"[eval] run {i}: {len(defects)} defects, "
                f"recall {score.detection_rate:.0%}, "
                f"precision {score.claim_precision:.0%}, "
                f"confirmed FP {score.false_positive_rate:.0%}"
            )
    if args.clean:
        report = render_clean_results(scores)
        report_name = "results-clean-control.md"
    else:
        report = render_results(scores)
        report_name = "results-metrics.md"
    report += "\n\n## Coverage per run\n\n" + "\n".join(coverage_lines) + "\n"
    (out / report_name).write_text(report)
    print(f"\n{report}\n\nwritten to {out}/{report_name}")


def cmd_injection(args: argparse.Namespace) -> None:
    out = _out_dir()
    results = []
    for variant in VARIANTS:
        print(f"[eval] injection variant: {variant.name}")
        results.append(run_variant(variant))
    print("[eval] injection variant: response-injection (triage)")
    response_results = [run_response_variant()]
    report = render_injection_results(results, response_results)
    (out / "results-injection.md").write_text(report)
    print(f"\n{report}\n\nwritten to {out}/results-injection.md")


def main() -> None:
    parser = argparse.ArgumentParser(prog="eval", description="QE agents eval harness")
    sub = parser.add_subparsers(dest="command", required=True)
    metrics = sub.add_parser("metrics", help="score repeated pipeline runs against bugs.yaml")
    metrics.add_argument("--runs", type=int, default=3)
    metrics.add_argument(
        "--clean",
        action="store_true",
        help="negative control: run against a bug-free SUT (SUT_CLEAN=1, "
        "flakiness off); every filed real/flaky defect is a false alarm "
        "unless its repro reproduces",
    )
    metrics.set_defaults(func=cmd_metrics)
    injection = sub.add_parser("injection", help="poisoned-artifact resistance probes")
    injection.set_defaults(func=cmd_injection)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
