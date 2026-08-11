"""Render the final run outputs: report.md for humans, defects.json for the
rest of the SDLC (CI gates, issue trackers). Also home of the coverage
accounting shared with the eval harness."""

import json
import re
from pathlib import Path

from qe_agent.state import QEState


def _norm_op(text: str) -> str:
    """'get /campaigns/{campaign_id}/' -> 'GET /campaigns/{id}'."""
    parts = text.strip().split(None, 1)
    if len(parts) != 2:
        return text.strip().upper()
    method, path = parts
    path = re.sub(r"\{[^}]*\}", "{id}", path.rstrip("/"))
    return f"{method.upper()} {path}"


def scenario_funnel(state: QEState) -> list[dict]:
    """Per-scenario fate: a scenario silently losing its test is a coverage
    hole nobody sees unless it is accounted for explicitly."""
    plan = state.get("plan")
    if not plan:
        return []
    execution = state.get("execution")
    generated = {t.scenario_id for t in state.get("generated_tests") or []}
    rejected = {r["scenario_id"] for r in state.get("rejected_tests") or []}
    excluded = set(state.get("excluded_scenarios") or [])
    executed = {r.scenario_id for r in (execution.results if execution else []) if r.scenario_id}

    rows = []
    for s in plan.scenarios:
        if s.id in executed:
            status = "executed"
        elif s.id in excluded:
            status = "excluded by reviewer"
        elif s.id in rejected:
            status = "rejected by static check"
        elif s.id in generated:
            status = "generated, not executed"
        else:
            status = "not generated"
        rows.append({"id": s.id, "title": s.title, "status": status})
    return rows


def operation_coverage(state: QEState) -> tuple[list[str], list[str]]:
    """(covered, uncovered) spec operations, judged by EXECUTED scenarios —
    a planned-but-dropped scenario earns no coverage credit."""
    model = state.get("system_model")
    plan = state.get("plan")
    if not model or not plan:
        return [], []
    spec_ops = {_norm_op(f"{e.method} {e.path}") for e in model.endpoints}
    executed = {row["id"] for row in scenario_funnel(state) if row["status"] == "executed"}
    exercised = {_norm_op(op) for s in plan.scenarios if s.id in executed for op in s.endpoints}
    return sorted(spec_ops & exercised), sorted(spec_ops - exercised)


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

    funnel = scenario_funnel(state)
    if funnel:
        covered, uncovered = operation_coverage(state)
        executed_n = sum(1 for row in funnel if row["status"] == "executed")
        lines += ["## Coverage", ""]
        if covered or uncovered:
            lines += [
                f"- spec operations exercised by executed tests: "
                f"{len(covered)}/{len(covered) + len(uncovered)}"
            ]
            if uncovered:
                lines += ["- **uncovered operations:**"]
                lines += [f"  - {op}" for op in uncovered]
        else:
            lines += ["- operation coverage unavailable (planner listed no endpoints)"]
        lines += [f"- scenario funnel: {executed_n}/{len(funnel)} planned scenarios executed"]
        dropped = [row for row in funnel if row["status"] != "executed"]
        if dropped:
            lines += ["- **scenarios that did not run:**"]
            lines += [f"  - {row['id']} ({row['status']}): {row['title']}" for row in dropped]
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
