"""CLI: qe run [--auto]

Everything runs against Docker: the SUT is started as a container (published
port for the host-side grounding fetch, internal-only network for the test
runner), and generated tests execute in the confined runner container.

Two HITL interrupts are surfaced on stdin:
1. ambiguity resolution (answer or accept the planner's assumption)
2. the test review stage gate — approve / exclude / edit on disk /
   send back for regeneration with feedback / abort

Exit codes (SDLC integration): 0 = no real defects, 2 = real defects found,
3 = run aborted at the review gate.
"""

import argparse
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

from langgraph.types import Command

from qe_agent import config, sandbox
from qe_agent.graph import build_graph


def _start_sut() -> None:
    port = urlparse(config.sut_base_url()).port or 8000
    sandbox.ensure_infra()
    sandbox.start_sut_container(host_port=port)  # blocks until /health is up


def _resolve_ambiguities(payload: dict) -> dict:
    print("\n=== Ambiguities surfaced by the planner ===")
    answers = {}
    for item in payload["items"]:
        print(f"\n[{item['id']}] {item['question']}")
        print(f"  context:    {item['context']}")
        print(f"  assumption: {item['assumption']}")
        try:
            reply = input("  your answer (empty = accept assumption): ").strip()
        except EOFError:
            reply = ""
        answers[item["id"]] = reply or f"accepted assumption: {item['assumption']}"
    return answers


def _print_review_table(files: list[dict], excluded: set[str]) -> None:
    for i, f in enumerate(files, 1):
        mark = "EXCLUDED" if f["file_name"] in excluded else ""
        print(
            f"  [{i}] {f['file_name']}  "
            f"(scenario {f['scenario_id']}, {len(f['code'].splitlines())} lines) {mark}"
        )


def _review_tests(payload: dict) -> dict:
    """Interactive review loop. Only the final decision resumes the graph."""
    files = payload["files"]
    excluded: set[str] = set()
    review_dir = Path(payload["run_dir"]) / "review"

    print(f"\n=== Test review gate (regeneration round {payload['revision_round']}) ===")
    _print_review_table(files, excluded)
    for r in payload.get("rejected_by_static_check", []):
        print(f"  REJECTED by static check: {r['file_name']}: {'; '.join(r['violations'])}")
    print(
        "\ncommands: a=approve & run | v N=view code | x N=toggle exclude | "
        "e=edit on disk | r N <feedback>=regenerate | q=abort"
    )

    while True:
        try:
            raw = input("review> ").strip()
        except EOFError:
            return {"action": "abort"}
        if not raw:
            continue
        cmd, _, rest = raw.partition(" ")
        cmd = cmd.lower()

        if cmd == "a":
            return {"action": "approve", "excluded": sorted(excluded)}
        if cmd == "q":
            return {"action": "abort"}
        if cmd == "v" and rest.isdigit() and 1 <= int(rest) <= len(files):
            f = files[int(rest) - 1]
            print(f"\n--- {f['file_name']} ---\n{f['code']}\n--- end ---")
            continue
        if cmd == "x" and rest.isdigit() and 1 <= int(rest) <= len(files):
            name = files[int(rest) - 1]["file_name"]
            excluded.symmetric_difference_update({name})
            _print_review_table(files, excluded)
            continue
        if cmd == "e":
            review_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                (review_dir / f["file_name"]).write_text(f["code"])
            input(f"files written to {review_dir} — edit them, then press Enter... ")
            changed = {}
            for f in files:
                new_code = (review_dir / f["file_name"]).read_text()
                if new_code != f["code"]:
                    changed[f["file_name"]] = new_code
            if not changed:
                print("no changes detected")
                continue
            print(f"submitting {len(changed)} edited file(s) for re-validation")
            return {"action": "edited", "files": changed}
        if cmd == "r":
            n, _, feedback = rest.partition(" ")
            if n.isdigit() and 1 <= int(n) <= len(files) and feedback.strip():
                scenario_id = files[int(n) - 1]["scenario_id"]
                return {"action": "revise", "feedback": {scenario_id: feedback.strip()}}
        print("unrecognized command")


def _print_summary(result: dict) -> None:
    print("\n=== Run complete ===")
    execution = result.get("execution")
    if execution:
        passed = sum(1 for r in execution.results if r.final_outcome == "passed")
        print(f"tests: {passed}/{len(execution.results)} passed (after retries)")
    defects = result.get("defects") or []
    print(f"defects: {len(defects)}")
    for d in defects:
        print(
            f"  {d.id} [{d.classification}/{d.severity}/{d.priority}] {d.title} -> {d.suspected_owner}"
        )
    if result.get("report_path"):
        print(f"report: {result['report_path']}")
    if result.get("defects_json_path"):
        print(f"defects.json: {result['defects_json_path']}")


def _exit_code(result: dict) -> int:
    if result.get("review_action") == "abort":
        return 3
    defects = result.get("defects") or []
    return 2 if any(d.classification == "real" for d in defects) else 0


def cmd_run(args: argparse.Namespace) -> int:
    _start_sut()
    graph = build_graph()
    thread = {"configurable": {"thread_id": uuid.uuid4().hex}}
    try:
        print(f"grounding on live spec: {config.sut_base_url()}/openapi.json")
        result = graph.invoke({"auto_approve": args.auto}, thread)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            if payload.get("type") == "ambiguities":
                resume = _resolve_ambiguities(payload)
            elif payload.get("type") == "test_review":
                resume = _review_tests(payload)
            else:
                raise RuntimeError(f"unknown interrupt payload: {payload}")
            result = graph.invoke(Command(resume=resume), thread)
    finally:
        sandbox.stop_sut_container()
    _print_summary(result)
    return _exit_code(result)


def main() -> None:
    parser = argparse.ArgumentParser(prog="qe", description="QE Agents pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="ground on the live OpenAPI spec, then plan→generate→execute→triage"
    )
    run.add_argument(
        "--auto", action="store_true", help="skip HITL gates (assumptions + auto-approve)"
    )
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
