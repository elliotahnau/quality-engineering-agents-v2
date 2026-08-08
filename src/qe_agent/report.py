"""Render the final run outputs: report.md for humans, defects.json for the
rest of the SDLC (CI gates, issue trackers)."""

import json
from pathlib import Path

from qe_agent.state import QEState


def render_report(state: QEState) -> str:
    plan = state.get("plan")
    execution = state.get("execution")
    defects = state.get("defects") or []
    lines: list[str] = ["# QE Agents run report", ""]

    warnings = state.get("injection_warnings") or []
    if warnings:
        lines += ["## ⚠️ Spec security warnings (possible prompt injection)", ""]
        lines += [f"- {w}" for w in warnings]
        lines += [""]

    model = state.get("system_model")
    if model:
        lines += ["## Grounding (live OpenAPI spec only)", "", model.summary, ""]
        lines += [f"- endpoints analyzed: {len(model.endpoints)}"]
        lines += [f"- risk areas: {'; '.join(model.risk_areas)}"]
        if model.inferred_invariants:
            lines += ["- inferred invariants (not documented, assumed by convention):"]
            lines += [f"  - {i}" for i in model.inferred_invariants]
        lines += [""]

    if plan:
        lines += ["## Test plan", "", plan.summary, ""]
        lines += [f"- scenarios: {len(plan.scenarios)}"]
        lines += [f"- entry criteria: {'; '.join(plan.entry_criteria)}"]
        lines += [f"- exit criteria: {'; '.join(plan.exit_criteria)}", ""]
        lines += ["| id | risk | types | title | basis |", "|---|---|---|---|---|"]
        lines += [
            f"| {s.id} | {s.risk.value} | {', '.join(t.value for t in s.test_types)} "
            f"| {s.title} | {s.basis[:90]} |"
            for s in plan.scenarios
        ]
        lines += [""]
        if plan.ambiguities:
            answers = state.get("human_answers") or {}
            lines += ["### Ambiguities surfaced (not silently guessed)", ""]
            for amb in plan.ambiguities:
                lines += [
                    f"- **{amb.id}** {amb.question}",
                    f"  - assumption: {amb.assumption}",
                    f"  - resolution: {answers.get(amb.id, '(unanswered)')}",
                ]
            lines += [""]

    rejected = state.get("rejected_tests") or []
    generated = state.get("generated_tests") or []
    lines += ["## Generation & review", ""]
    lines += [f"- test modules after review: {len(generated)}"]
    if state.get("revision_round"):
        lines += [f"- reviewer-requested regeneration rounds: {state['revision_round']}"]
    if rejected:
        lines += [f"- **rejected by static safety check: {len(rejected)}**"]
        for r in rejected:
            lines += [f"  - {r['file_name']}: {'; '.join(r['violations'])}"]
    lines += [""]

    if execution:
        passed = sum(1 for r in execution.results if r.final_outcome == "passed")
        lines += ["## Execution (Executor)", ""]
        lines += [
            f"- sandbox: {execution.sandbox}, retries per failing test: {execution.retries}",
            f"- {passed}/{len(execution.results)} passed (final outcome), "
            f"{execution.duration_seconds}s",
            "",
            "| test | attempts | final |",
            "|---|---|---|",
        ]
        lines += [
            f"| {r.test_id} | {' → '.join(r.attempts)} | {r.final_outcome} |"
            for r in execution.results
        ]
        lines += [""]

    lines += ["## Defects (Auditor)", ""]
    if not defects:
        lines += ["No defects. 🎉", ""]
    for d in defects:
        lines += [
            f"### {d.id} [{d.severity}/{d.priority}] {d.title}",
            "",
            f"- classification: **{d.classification}**",
            f"- endpoint: `{d.endpoint}`",
            f"- spec basis: {'; '.join(d.spec_refs) or '-'}",
            f"- scenarios: {', '.join(d.scenario_ids)} | tests: {', '.join(d.test_ids)}",
            f"- suspected owner: {d.suspected_owner}",
            f"- evidence: {d.evidence}",
            f"- root cause hypothesis: {d.root_cause_hypothesis}",
            "",
        ]
    return "\n".join(lines)


def node_report(state: QEState) -> dict:
    run_dir = Path(state["run_dir"])
    report_path = run_dir / "report.md"
    report_path.write_text(render_report(state))

    defects = state.get("defects") or []
    defects_json_path = run_dir / "defects.json"
    defects_json_path.write_text(
        json.dumps([d.model_dump() for d in defects], indent=2, ensure_ascii=False)
    )
    return {"report_path": str(report_path), "defects_json_path": str(defects_json_path)}
