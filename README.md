# QE Agents

A LangGraph pipeline of QE agents that grounds itself on a **running API's live
OpenAPI spec** — never its source code — then plans, generates, executes and
triages its way to defects an engineer can act on.

The spec is the intended contract, the runtime responses are the actual
behavior, and every defect is a gap between the two.

```
GET /openapi.json ─▶ ground ─▶ plan ─▶ [HITL: ambiguities] ─▶ generate ─▶ syntax check
                                                    ▲                        │
                                    (revise: feedback)                       ▼
report ◀── triage (Auditor) ◀── execute (Executor) ◀── approve ─[HITL: test review]
   ▲                                                              │        │
   └────────────────────────── abort ─────────────────────────────┘   edited┘
                                                            (re-validate, back to review)
```

- **Grounding & Analysis** — builds a system model from the live spec,
  separating rules the spec *documents* from invariants it merely implies
  (marked `INFERRED`).
- **Test Planning** — 12–16 risk-ordered scenarios across boundary, negative,
  5xx-hunting, state transition, idempotency and concurrency. Where the spec
  is silent the planner raises a question instead of guessing.
- **Test Generation** — one pytest module per scenario; only code that parses
  reaches the review gate.
- **HITL Review Gate** — view, exclude, edit on disk, or send a scenario back
  with feedback (which loops the graph back to generation). Edited code is
  re-validated; a human edit is not automatically trusted.
- **Execution (Executor)** — runs in a **Docker sandbox**: the runner container
  sits on an internal-only network with no external egress, a read-only
  rootfs, dropped capabilities, non-root, and pid/memory/CPU limits. Its
  reachable world is exactly the SUT. Failures are retried, and the attempt
  history becomes flaky evidence.
- **Defect Triaging (Auditor)** — clusters failures by signature, classifies
  real / flaky / test-bug (a deterministic pass-on-retry rule overrides the
  model), and assigns severity, priority, a black-box root-cause hypothesis
  and an owner.

The system under test in `sut/` is a small ad-ops campaign API carrying 7
planted bugs and 2 reproducible flaky faults, all labeled in `sut/bugs.yaml`.
That manifest is eval-only ground truth — the agents read nothing but the spec.

**Design rationale, trade-offs and evaluation results:**
[`docs/design.md`](docs/design.md) · [`eval/RESULTS.md`](eval/RESULTS.md)

## Setup

Requires Python 3.12+ and **Docker** (the sandbox is not optional).

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
cp .env.example .env   # set GEMINI_API_KEY
```

## Run

```bash
# interactive demo: answer ambiguities, then review/edit/regenerate tests
.venv/bin/qe run

# non-interactive: proceed on the planner's assumptions, auto-approve tests
.venv/bin/qe run --auto
```

`qe run` manages the SUT container and sandbox network itself (the first run
builds the sandbox image, ~1 minute).

Artifacts land in `.qe_runs/<timestamp>/`: `openapi.json` (the exact grounding
input), the generated tests, `report.md`, and `defects.json`.

**Exit codes** for CI: `0` no real defects, `2` real defects found, `3`
aborted at the review gate.

## Evaluate

```bash
.venv/bin/python -m eval metrics --runs 3   # score runs against sut/bugs.yaml
.venv/bin/python -m eval injection          # poisoned-spec resistance probes
```

## Test / Lint

```bash
.venv/bin/pytest          # unit tests; no LLM, network or Docker needed
.venv/bin/ruff check .
.venv/bin/black --check .
```

## Layout

| path | what it is |
|---|---|
| `src/qe_agent/graph.py` | LangGraph wiring: two HITL interrupts and the revise loop |
| `src/qe_agent/stages/` | grounding·planning / generation / review·execution / triage |
| `src/qe_agent/schemas.py` | typed contracts: model-produced vs fact-produced state |
| `src/qe_agent/security.py` | prompt-injection defense (spotlighting + scanner) |
| `src/qe_agent/sandbox.py` | Docker sandbox: internal network, isolation flags, JUnit parsing |
| `docker/Dockerfile.sandbox` | shared image for the SUT and runner containers |
| `sut/` | system under test + `bugs.yaml` ground truth |
| `eval/` | eval harness and results |
| `docs/design.md` | architecture, trade-offs, evaluation, what's next |
