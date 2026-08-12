# quality-engineering-agents-v2

A LangGraph multi-agent Quality Engineering pipeline that tests an HTTP API it
has never seen. The agents ground themselves on the **live OpenAPI spec only**
(never the source code), plan risk-based scenarios, generate pytest modules,
run them in a confined Docker sandbox, and triage the failures into defects
with severity, priority, and a suspected owner.

The system under test (SUT) is an ad-operations campaign API shipped in this
repo with **8 planted bugs and 2 planted flaky faults**, labeled in
`sut/bugs.yaml`. That label file is the ground truth the eval harness scores
against — the agents never read it. (The 8th bug was found by the pipeline
itself and promoted from an eval finding to ground truth.)

Architecture, framework/model trade-offs, evaluation results, and what we'd
build next: [`docs/DESIGN.md`](docs/DESIGN.md).

```text
ground → plan → [HITL ambiguity gate] → generate → static_check
      → [HITL review gate] ─ approve → execute → triage → report
                           ├ revise  → generate       (feedback loop)
                           ├ edited  → static_check   (re-validate human edits)
                           └ abort   → report
```

## Requirements

| | |
|---|---|
| Python | 3.12+ |
| Docker | running daemon (the SUT and the test runner are containers) |
| LLM key | `GEMINI_API_KEY` (default) or `OPENAI_API_KEY` |

## Clone and test

```bash
git clone https://github.com/elliotahnau/quality-engineering-agents-v2.git
cd quality-engineering-agents-v2

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 1. Offline checks (no API key, no Docker)

The unit suite mocks the LLM and never starts a container, so this works
immediately after install:

```bash
pytest                             # unit tests
ruff check . && black --check .    # lint + format, same as CI
```

### 2. Configure an LLM provider

```bash
cp .env.example .env
```

Then edit `.env`:

```ini
LLM_PROVIDER=gemini            # or: openai
LLM_MODEL=gemini-3.6-flash     # or e.g. gpt-5.4-mini for openai
GEMINI_API_KEY=your-key-here
# OPENAI_API_KEY=

SUT_BASE_URL=http://127.0.0.1:8000
```

### 3. Run the full pipeline

Make sure Docker is running, then:

```bash
qe run              # interactive: stops at both HITL gates
qe run --auto       # non-interactive: accepts the planner's assumptions,
                    # auto-approves the review gate (used by the eval harness)
```

The first invocation builds the `qe-sandbox:latest` image (a couple of minutes,
once). Everything after that is fast. The SUT container is started and torn
down for you — you do not need to launch it separately.

`qe run` stops at two human-in-the-loop gates:

**Ambiguity gate** — the planner lists behaviors the spec leaves unspecified
(e.g. whether a report date range is inclusive). Answer, or press Enter to
accept its stated assumption.

**Review gate** — inspect the generated tests before anything executes:

| command | effect |
|---|---|
| `a` | approve and run |
| `v N` | view the code of test N |
| `x N` | toggle excluding test N from the run |
| `e` | dump the tests to disk, edit them, re-validate on return |
| `r N <feedback>` | send test N back for regeneration with your feedback |
| `q` | abort the run |

### 4. Read the output

Each run writes to `.qe_runs/<timestamp>-<run-id>/`:

| file | contents |
|---|---|
| `report.md` | human report: grounding summary, test plan, coverage (scenario funnel + spec operations exercised), execution results, defects |
| `defects.json` | machine-readable defects (with a replayable single-request repro where possible) for CI gates / issue trackers |
| `openapi.json` | the exact spec the agents grounded on |
| `tests/` | the generated pytest modules that ran |
| `out/junit.xml` | raw JUnit report from the sandbox |

Exit codes make the pipeline usable as an SDLC gate:

| code | meaning |
|---|---|
| `0` | no real defects |
| `2` | real defects found |
| `3` | run aborted at the review gate |

A good run finds most of the 8 planted bugs — try cross-checking `report.md`
against `sut/bugs.yaml` after the fact.

## Poking at the SUT directly

Useful for understanding what the agents are up against:

```bash
python -m sut --port 8000
```

Then open <http://127.0.0.1:8000/docs>, or fetch the spec the agents see at
`/openapi.json`. Nine endpoints under `/campaigns` and `/reports`.

Flakiness is counter-based rather than random, so runs stay reproducible —
every Nth `resume` call returns 503, every (N+2)th `GET /campaigns` returns a
stale empty list. Tune with `SUT_FLAKY_EVERY=5` (`0` disables it).
`SUT_CLEAN=1` switches every planted bug off — the eval's negative control.

## Eval harness — how we test the tester

```bash
python -m eval metrics --runs 3   # score repeated runs against bugs.yaml
python -m eval metrics --clean    # negative control: bug-free SUT
python -m eval injection          # poisoned-artifact resistance probes
```

`metrics` starts a fresh SUT container per run (so the flaky counters fire at
reproducible points) and reports recall over real bugs and flaky faults, a
classification confusion matrix, defect precision, false-positive rate,
duplicate filings, and operation coverage per run — variance shown, not
averaged away. Defects carry a structured single-request repro that the
harness **replays against the live SUT**, so an unmatched defect is
automatically judged: refuted → confirmed false alarm, reproduced → genuine
behavior missing from the label set, no repro → unverified claim.

`--clean` runs the same pipeline against a bug-free SUT (`SUT_CLEAN=1`,
flakiness off): with nothing planted, every filed defect is a false alarm
unless its repro reproduces — the "does it cry wolf on a healthy system?"
control.

`injection` plants hostile directives in both SUT-controlled channels — the
OpenAPI spec (grounding/planning) and a failure message quoting a response
body (triage) — and checks that the plan is not suppressed, risk levels are
not flattened, defects are not suppressed or reclassified, and a canary
string is not echoed. All write to `.qe_runs/eval-<timestamp>/`.


## Security model

- **Ingested artifacts are untrusted data.** The fetched OpenAPI spec is
  wrapped in unique boundary markers and scanned for instruction-like content;
  warnings are shown to the human **at both HITL gates**, before anything is
  approved, and again in `report.md` ([security.py](src/qe_agent/security.py)).
- **SUT responses are the second untrusted channel.** Failure messages and
  tracebacks quote response bodies, so triage quarantines them behind the same
  boundary-marker scheme and scans them too — a hostile SUT asking the triager
  to bury its own defects is an eval-probed attack, not an open hole.
- **The sandbox is the real boundary for generated code**, not the AST check.
  The runner container is on an internal-only Docker network with a read-only
  rootfs, all capabilities dropped, `no-new-privileges`, and pid/memory/CPU
  caps; test sources are mounted read-only and only a per-run `out/` directory
  is writable ([sandbox.py](src/qe_agent/sandbox.py)).
- The AST static check is a generation-quality fast-fail that keeps
  unparseable modules away from the human reviewer — not a security device.

## Layout

| path | |
|---|---|
| [src/qe_agent/graph.py](src/qe_agent/graph.py) | LangGraph wiring |
| [src/qe_agent/stages/](src/qe_agent/stages/) | planning, generation, execution, triage |
| [src/qe_agent/sandbox.py](src/qe_agent/sandbox.py) | Docker isolation contract, JUnit parsing |
| [src/qe_agent/security.py](src/qe_agent/security.py) | injection defense, AST check |
| [src/qe_agent/cli.py](src/qe_agent/cli.py) | `qe run`, the HITL prompts |
| [sut/](sut/) | the SUT, its planted bugs, and the ownership map |
| [eval/](eval/) | metric + injection harnesses, defect replay verifier |
| [tests/](tests/) | offline unit tests (LLM mocked, no Docker) |

## Troubleshooting

**`GEMINI_API_KEY is not set`** — `.env` is missing or the key is empty.

**`SUT container did not become healthy`** — the Docker daemon is not running,
or port 8000 is already taken. Free the port or change `SUT_BASE_URL` in
`.env`.

**Rebuild the sandbox image** after editing `docker/Dockerfile.sandbox`:

```bash
docker rmi qe-sandbox:latest      # next run rebuilds it
```

**Stale containers** after a hard interrupt:

```bash
docker rm -f qe-sut $(docker ps -aq --filter "name=qe-runner-")
```
