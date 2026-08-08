"""Eval entry point.

  python -m eval metrics [--runs 3]    # repeated full pipeline runs, scored
  python -m eval injection             # poisoned-spec resistance probes

Both need GEMINI_API_KEY; `metrics` additionally needs Docker (SUT container
is started fresh per run so flaky counters stay reproducible).
Results are written to .qe_runs/eval-<timestamp>/.
"""

import argparse
import time
from pathlib import Path
from urllib.parse import urlparse

from eval.harness import dump_defects, render_results, run_pipeline_once, score_run
from eval.injection import VARIANTS, render_injection_results, run_variant
from qe_agent import config, sandbox
from qe_agent.graph import build_graph


def _out_dir() -> Path:
    out = config.RUNS_DIR / f"eval-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def cmd_metrics(args: argparse.Namespace) -> None:
    out = _out_dir()
    port = urlparse(config.sut_base_url()).port or 8000
    sandbox.ensure_infra()
    scores = []
    for i in range(1, args.runs + 1):
        print(f"[eval] run {i}/{args.runs}: starting fresh SUT container...")
        sandbox.start_sut_container(host_port=port)
        try:
            defects, _ = run_pipeline_once(build_graph())
        finally:
            sandbox.stop_sut_container()
        (out / f"run-{i}-defects.json").write_text(dump_defects(defects))
        score = score_run(defects)
        scores.append(score)
        print(
            f"[eval] run {i}: {len(defects)} defects, "
            f"detection {score.detection_rate:.0%}, FP {score.false_positive_rate:.0%}"
        )
    report = render_results(scores)
    (out / "results-metrics.md").write_text(report)
    print(f"\n{report}\n\nwritten to {out}/results-metrics.md")


def cmd_injection(args: argparse.Namespace) -> None:
    out = _out_dir()
    results = []
    for variant in VARIANTS:
        print(f"[eval] injection variant: {variant.name}")
        results.append(run_variant(variant))
    report = render_injection_results(results)
    (out / "results-injection.md").write_text(report)
    print(f"\n{report}\n\nwritten to {out}/results-injection.md")


def main() -> None:
    parser = argparse.ArgumentParser(prog="eval", description="QE agents eval harness")
    sub = parser.add_subparsers(dest="command", required=True)
    metrics = sub.add_parser("metrics", help="score repeated pipeline runs against bugs.yaml")
    metrics.add_argument("--runs", type=int, default=3)
    metrics.set_defaults(func=cmd_metrics)
    injection = sub.add_parser("injection", help="poisoned-spec resistance probes")
    injection.set_defaults(func=cmd_injection)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
