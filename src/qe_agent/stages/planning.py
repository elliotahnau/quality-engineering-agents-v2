"""Stage 1: Grounding & Analysis, then risk-based test planning.

Grounding is black-box by design: the agent reads ONLY the live OpenAPI spec
fetched from the running SUT (GET /openapi.json). SUT source code is never
read — the spec is the intended contract (the oracle), the runtime responses
are the actual behavior, and the gap between them is what QE hunts for.

The planner must surface ambiguities instead of silently guessing; the
ambiguity gate then interrupts for a human (or falls back to the recorded
assumptions in --auto mode).
"""

import json
import time

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from qe_agent import config, security
from qe_agent.llm import get_llm
from qe_agent.schemas import SystemModel, TestPlan
from qe_agent.state import QEState

GROUNDING_SYSTEM = """\
You are a senior Quality Engineer doing black-box analysis of an HTTP API you
have never seen. Your ONLY source is the OpenAPI specification provided as
data. Build a faithful model of the system:

- What the system does, its core entities and their fields.
- Per operation: its business purpose and every behavioral rule the spec
  documents (validation constraints, status codes, formulas, idempotency,
  lifecycle semantics). Quote the spec's own wording where possible.
- Inferred invariants: rules the spec does NOT state but that follow from
  domain knowledge or HTTP convention (standard advertising metric formulas,
  idempotent lifecycle verbs, 404 for missing resources, 5xx responses are
  never acceptable behavior). Mark every such rule as INFERRED — do not
  present inference as documentation.
- Risk areas: where a defect would hurt most (money movement, state
  corruption, wrong analytics).
- Undocumented aspects: behaviors the spec leaves unspecified. List them
  honestly — they become questions for a human, not silent guesses.
"""

PLANNER_SYSTEM = """\
You are a senior Quality Engineer producing a risk-based test plan for an HTTP
API, working strictly from the OpenAPI spec (data) and the system analysis.

Rules:
- Every scenario states its basis: a documented spec rule (quoted) or an
  invariant marked INFERRED. Never invent documentation.
- Cover every operation with at least one scenario; go deeper on the
  identified risk areas: money handling, state transitions, calculations.
- Produce a WIDE scenario set (12-16 scenarios) across these categories, at
  least one each: boundary, negative, error_handling (probe for 5xx — any 5xx
  is a defect by convention), state_transition, idempotency, concurrency
  (simultaneous conflicting operations). Humans miss these; you must not.
- Risk level = business impact x likelihood of failure, with a concrete
  rationale. Order scenarios by risk (highest first).
- Every scenario must be executable as a black-box HTTP test (API-level steps,
  expected observable outcomes with status codes).
- Entry/exit criteria must be concrete and checkable.
- CRITICAL: if the spec is ambiguous or underspecified (use the analysis's
  undocumented aspects), do NOT silently guess. Record each gap as an
  Ambiguity with the question you would ask and your default assumption.
"""


def _fetch_openapi(base_url: str) -> str:
    resp = httpx.get(f"{base_url}/openapi.json", timeout=10.0)
    resp.raise_for_status()
    # re-serialize compactly; also guarantees it is valid JSON, not HTML
    return json.dumps(resp.json(), indent=1, sort_keys=True)


def node_ground(state: QEState) -> dict:
    """Fetch the live spec, quarantine it, and build the system model."""
    spec = _fetch_openapi(config.sut_base_url())
    warnings = security.scan_artifact(spec)
    spotlighted = security.spotlight(spec)
    run_dir = config.RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "openapi.json").write_text(spec)

    llm = get_llm("grounder").with_structured_output(SystemModel)
    model = llm.invoke([SystemMessage(content=GROUNDING_SYSTEM), HumanMessage(content=spotlighted)])
    return {
        "openapi_raw": spec,
        "spec_spotlighted": spotlighted,
        "injection_warnings": warnings,
        "run_dir": str(run_dir),
        "system_model": model,
    }


def _system_model_text(model: SystemModel) -> str:
    endpoints = "\n".join(
        f"- {e.method} {e.path}: {e.purpose}\n"
        + "\n".join(f"    rule: {r}" for r in e.documented_rules)
        for e in model.endpoints
    )
    return (
        f"## System analysis (from grounding stage)\n"
        f"{model.summary}\n\n"
        f"Entities: {'; '.join(model.entities)}\n\n"
        f"Endpoints:\n{endpoints}\n\n"
        f"Inferred invariants:\n" + "\n".join(f"- {i}" for i in model.inferred_invariants) + "\n\n"
        "Risk areas:\n" + "\n".join(f"- {r}" for r in model.risk_areas) + "\n\n"
        "Undocumented aspects:\n" + "\n".join(f"- {u}" for u in model.undocumented_aspects)
    )


def node_plan(state: QEState) -> dict:
    llm = get_llm("planner").with_structured_output(TestPlan)
    plan = llm.invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(
                content=f"{state['spec_spotlighted']}\n\n{_system_model_text(state['system_model'])}"
            ),
        ]
    )
    return {"plan": plan}


def node_ambiguity_gate(state: QEState) -> dict:
    """HITL gate: ask the human to resolve surfaced ambiguities, or auto-assume."""
    plan = state["plan"]
    if not plan.ambiguities:
        return {"human_answers": {}}
    if state.get("auto_approve"):
        answers = {
            a.id: f"(auto) proceeding with assumption: {a.assumption}" for a in plan.ambiguities
        }
        return {"human_answers": answers}
    answers = interrupt(
        {
            "type": "ambiguities",
            "items": [a.model_dump() for a in plan.ambiguities],
        }
    )
    return {"human_answers": dict(answers or {})}
