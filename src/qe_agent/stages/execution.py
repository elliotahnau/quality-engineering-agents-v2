"""Stage 3: HITL review gate, then sandboxed execution with retries.

The review gate is a full stage gate, not a yes/no prompt: the QE reviewer can
approve (optionally excluding tests), hand back per-scenario feedback for
regeneration, submit edited code, or abort. Edited and regenerated code always
passes back through the AST static check before it can run.

Failed tests are rerun (RETRIES extra attempts); the full attempt history per
test is recorded because pass-on-retry is the triage stage's primary evidence
for flaky classification.
"""

import time
from pathlib import Path

from langgraph.types import interrupt

from qe_agent.sandbox import DockerRunner, write_test_dir
from qe_agent.schemas import CaseResult, ExecutionReport
from qe_agent.state import QEState

RETRIES = 2


def node_review_gate(state: QEState) -> dict:
    """HITL stage gate over the generated test cases.

    Resume payload shapes:
      {"action": "approve", "excluded": ["test_x.py", ...]}
      {"action": "revise", "feedback": {"TS-003": "also cover zero"}}
      {"action": "edited", "files": {"test_x.py": "<new code>"}}
      {"action": "abort"}
    """
    if state.get("auto_approve"):
        return {"review_action": "approve"}

    decision = interrupt(
        {
            "type": "test_review",
            "files": [
                {"file_name": t.file_name, "scenario_id": t.scenario_id, "code": t.code}
                for t in state["generated_tests"]
            ],
            "rejected_by_static_check": state.get("rejected_tests", []),
            "revision_round": state.get("revision_round") or 0,
            "run_dir": state["run_dir"],
        }
    )
    action = decision.get("action", "abort")

    if action == "approve":
        excluded = set(decision.get("excluded") or [])
        kept = [t for t in state["generated_tests"] if t.file_name not in excluded]
        return {"review_action": "approve", "generated_tests": kept}

    if action == "revise":
        return {
            "review_action": "revise",
            "revision_feedback": dict(decision.get("feedback") or {}),
        }

    if action == "edited":
        new_files = dict(decision.get("files") or {})
        updated = [
            t.model_copy(update={"code": new_files[t.file_name]}) if t.file_name in new_files else t
            for t in state["generated_tests"]
        ]
        return {"review_action": "edited", "generated_tests": updated}

    return {"review_action": "abort"}


def route_after_review(state: QEState) -> str:
    return {
        "approve": "execute",
        "revise": "generate",
        "edited": "static_check",
    }.get(state.get("review_action", "abort"), "report")


def node_execute(state: QEState) -> dict:
    started = time.monotonic()
    run_dir = Path(state["run_dir"])
    files = {t.file_name: t.code for t in state["generated_tests"]}
    file_to_scenario = {Path(t.file_name).name: t.scenario_id for t in state["generated_tests"]}
    test_dir = write_test_dir(run_dir, files)

    runner = DockerRunner()
    first = runner.run(test_dir)

    results: dict[str, CaseResult] = {}
    for node_id, outcome in first.items():
        file_name = node_id.split("::")[0]
        results[node_id] = CaseResult(
            test_id=node_id,
            scenario_id=file_to_scenario.get(file_name),
            attempts=[outcome.outcome],
            message=outcome.message,
            traceback=outcome.traceback,
        )

    failing = [nid for nid, r in results.items() if r.attempts[-1] in ("failed", "error")]
    for _ in range(RETRIES):
        if not failing:
            break
        rerun = runner.run(test_dir, node_ids=failing)
        still_failing = []
        for node_id in failing:
            case = results[node_id]
            outcome = rerun.get(node_id)
            if outcome is None:
                # the runner did not report a test we asked it to rerun (crashed
                # container, missing report). Record it rather than dropping the
                # attempt — triage reads attempt history to judge flakiness.
                case.attempts.append("error")
                case.message = case.message or "no result returned by sandbox rerun"
                still_failing.append(node_id)
                continue
            case.attempts.append(outcome.outcome)
            if outcome.outcome in ("failed", "error"):
                case.message = outcome.message or case.message
                case.traceback = outcome.traceback or case.traceback
                still_failing.append(node_id)
        failing = still_failing

    report = ExecutionReport(
        results=sorted(results.values(), key=lambda r: r.test_id),
        retries=RETRIES,
        sandbox=runner.name,
        duration_seconds=round(time.monotonic() - started, 2),
    )
    return {"execution": report}
