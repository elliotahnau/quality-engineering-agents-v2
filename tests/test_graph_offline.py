"""Full graph wiring tests, fully offline: the spec fetch, LLMs and the
sandbox runner are faked, so these verify grounding state flow, the static-
check gate, the review stage gate (approve/exclude, revise loop, edited
re-validation, abort) and report rendering without an API key or a live SUT.
"""

from pathlib import Path

import pytest
from langgraph.types import Command

from qe_agent import config
from qe_agent.graph import build_graph
from qe_agent.sandbox import CaseOutcome
from qe_agent.schemas import (
    Ambiguity,
    Defect,
    EndpointUnderstanding,
    GeneratedTest,
    GeneratedTestBatch,
    RiskLevel,
    SystemModel,
    TestPlan,
    TestScenario,
    TestType,
    TriageResult,
)
from qe_agent.stages import execution as execution_stage
from qe_agent.stages import generation as generation_stage
from qe_agent.stages import planning as planning_stage
from qe_agent.stages import triage as triage_stage

GOOD_CODE_1 = (
    "import pytest\n\n\n"
    "def test_ts_001_rejects_negative_daily_budget(client):\n"
    "    resp = client.post('/campaigns', json={'daily_budget': -1})\n"
    "    assert resp.status_code == 422\n"
)
GOOD_CODE_2 = (
    "import pytest\n\n\n"
    "def test_ts_002_pause_is_idempotent(client):\n"
    "    resp = client.post('/campaigns/x/pause')\n"
    "    assert resp.status_code in (200, 404)\n"
)
BROKEN_CODE = "def test_ts_003_broken(client:\n    assert True\n"  # unparseable
EDITED_CODE_1 = GOOD_CODE_1.replace("== 422", "== 422  # reviewed")

FAKE_MODEL = SystemModel(
    summary="Ad campaign management API",
    entities=["Campaign: id, name, channel, budgets, dates, status"],
    endpoints=[
        EndpointUnderstanding(
            method="POST",
            path="/campaigns",
            purpose="create a campaign",
            documented_rules=["daily_budget must be greater than 0"],
        )
    ],
    inferred_invariants=["INFERRED: 5xx responses are never acceptable"],
    risk_areas=["budget validation"],
    undocumented_aspects=["maximum name length"],
)

FAKE_PLAN = TestPlan(
    summary="risk-based plan",
    in_scope=["campaign API"],
    out_of_scope=["auth"],
    entry_criteria=["SUT healthy"],
    exit_criteria=["all critical scenarios executed"],
    scenarios=[
        TestScenario(
            id="TS-001",
            title="daily_budget boundary validation",
            basis="POST /campaigns: 'daily_budget must be greater than 0'",
            risk=RiskLevel.high,
            risk_rationale="money handling",
            test_types=[TestType.boundary, TestType.negative],
            steps=["POST /campaigns with daily_budget=-1"],
            expected="422",
        ),
        TestScenario(
            id="TS-002",
            title="pause idempotency",
            basis="POST /campaigns/{id}/pause: 'idempotent'",
            risk=RiskLevel.medium,
            risk_rationale="state",
            test_types=[TestType.idempotency],
            steps=["pause twice"],
            expected="paused",
        ),
        TestScenario(
            id="TS-003",
            title="unknown resource error handling",
            basis="INFERRED: missing resources return 404, never 5xx",
            risk=RiskLevel.medium,
            risk_rationale="robustness",
            test_types=[TestType.error_handling],
            steps=["POST /campaigns with SQL fragment name"],
            expected="not 5xx",
        ),
    ],
    ambiguities=[
        Ambiguity(
            id="AMB-001",
            question="max name length?",
            context="POST /campaigns name field",
            assumption="no limit enforced",
        )
    ],
)

FAKE_BATCH = GeneratedTestBatch(
    tests=[
        GeneratedTest(scenario_id="TS-001", file_name="test_ts_001_budget.py", code=GOOD_CODE_1),
        GeneratedTest(scenario_id="TS-002", file_name="test_ts_002_pause.py", code=GOOD_CODE_2),
        GeneratedTest(scenario_id="TS-003", file_name="test_ts_003_broken.py", code=BROKEN_CODE),
    ]
)

FAKE_TRIAGE = TriageResult(
    defects=[
        Defect(
            id="DEF-001",
            title="negative daily_budget accepted",
            classification="real",
            severity="high",
            priority="P1",
            endpoint="POST /campaigns",
            scenario_ids=["TS-001"],
            test_ids=["test_ts_001_budget.py::test_ts_001_rejects_negative_daily_budget"],
            evidence="expected 422 per spec rule, observed 201",
            root_cause_hypothesis="missing validation on daily_budget",
            suspected_owner="campaign-core-team",
            spec_refs=["POST /campaigns: 'daily_budget must be greater than 0'"],
        )
    ]
)


class FakeLLM:
    def __init__(self, obj):
        self.obj = obj

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return self.obj


class FakeRunner:
    name = "fake"

    def __init__(self, timeout=300):
        pass

    def run(self, test_dir, node_ids=None):
        outcomes = {
            "test_ts_001_budget.py::test_ts_001_rejects_negative_daily_budget": CaseOutcome(
                outcome="failed", message="assert 201 == 422"
            ),
            "test_ts_002_pause.py::test_ts_002_pause_is_idempotent": CaseOutcome(outcome="passed"),
        }
        if node_ids is None:
            return outcomes
        return {nid: outcomes[nid] for nid in node_ids if nid in outcomes}


@pytest.fixture()
def offline_graph(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(planning_stage, "_fetch_openapi", lambda base_url: '{"openapi": "3.1.0"}')
    monkeypatch.setattr(
        planning_stage,
        "get_llm",
        lambda role="": FakeLLM(FAKE_MODEL if role == "grounder" else FAKE_PLAN),
    )
    monkeypatch.setattr(generation_stage, "get_llm", lambda role="": FakeLLM(FAKE_BATCH))
    monkeypatch.setattr(triage_stage, "get_llm", lambda role="": FakeLLM(FAKE_TRIAGE))
    monkeypatch.setattr(execution_stage, "DockerRunner", FakeRunner)
    return build_graph()


def test_full_graph_auto_mode(offline_graph):
    result = offline_graph.invoke({"auto_approve": True}, {"configurable": {"thread_id": "t1"}})

    # grounding produced the system model from the live spec only
    assert result["system_model"].summary.startswith("Ad campaign")
    assert result["openapi_raw"] == '{"openapi": "3.1.0"}'

    # ambiguity gate auto-answered with the assumption
    assert "AMB-001" in result["human_answers"]

    # syntax fast-fail dropped the unparseable module before the review gate
    names = [t.file_name for t in result["generated_tests"]]
    assert "test_ts_003_broken.py" not in names
    assert result["rejected_tests"][0]["file_name"] == "test_ts_003_broken.py"

    # failing test retried, defect kept as real; passing test untouched
    by_id = {r.test_id: r for r in result["execution"].results}
    assert by_id["test_ts_001_budget.py::test_ts_001_rejects_negative_daily_budget"].attempts == [
        "failed",
        "failed",
        "failed",
    ]
    assert len(result["defects"]) == 1

    report_text = Path(result["report_path"]).read_text()
    assert "DEF-001" in report_text
    assert "Grounding (live OpenAPI spec only)" in report_text
    assert Path(result["defects_json_path"]).read_text().count("DEF-001") == 1


def test_review_gate_revise_edit_approve_flow(offline_graph):
    thread = {"configurable": {"thread_id": "t2"}}
    result = offline_graph.invoke({}, thread)

    assert result["__interrupt__"][0].value["type"] == "ambiguities"
    result = offline_graph.invoke(Command(resume={"AMB-001": "cap at 100 chars"}), thread)

    # round 0: send TS-001 back with feedback -> generate runs again
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "test_review"
    assert payload["revision_round"] == 0
    assert payload["run_dir"]
    result = offline_graph.invoke(
        Command(resume={"action": "revise", "feedback": {"TS-001": "also cover zero"}}), thread
    )

    # round 1: submit a human edit -> re-validated by static check
    payload = result["__interrupt__"][0].value
    assert payload["revision_round"] == 1
    result = offline_graph.invoke(
        Command(resume={"action": "edited", "files": {"test_ts_001_budget.py": EDITED_CODE_1}}),
        thread,
    )

    # approve, excluding the pause test
    payload = result["__interrupt__"][0].value
    codes = {f["file_name"]: f["code"] for f in payload["files"]}
    assert codes["test_ts_001_budget.py"] == EDITED_CODE_1
    result = offline_graph.invoke(
        Command(resume={"action": "approve", "excluded": ["test_ts_002_pause.py"]}), thread
    )

    assert result["review_action"] == "approve"
    assert [t.file_name for t in result["generated_tests"]] == ["test_ts_001_budget.py"]
    assert result["generated_tests"][0].code == EDITED_CODE_1
    assert result["revision_round"] == 1
    assert result["defects"][0].id == "DEF-001"


def test_review_gate_keeps_module_order_across_a_revision(offline_graph):
    """The reviewer picks tests by number, so a regenerated module has to come
    back in the slot it left instead of at the bottom of the list — and a model
    that also returns scenarios nobody asked about must not duplicate the ones
    being kept."""
    thread = {"configurable": {"thread_id": "t5"}}
    result = offline_graph.invoke({}, thread)
    result = offline_graph.invoke(Command(resume={"AMB-001": "n/a"}), thread)

    before = [f["scenario_id"] for f in result["__interrupt__"][0].value["files"]]
    assert before == ["TS-001", "TS-002"]

    # the fake generator answers with the whole batch, standing in for a model
    # that returns more than the single scenario it was asked to redo
    result = offline_graph.invoke(
        Command(resume={"action": "revise", "feedback": {"TS-001": "cover zero too"}}), thread
    )
    after = [f["scenario_id"] for f in result["__interrupt__"][0].value["files"]]
    assert after == before


def test_review_gate_edited_code_must_pass_static_check(offline_graph):
    thread = {"configurable": {"thread_id": "t3"}}
    result = offline_graph.invoke({}, thread)
    result = offline_graph.invoke(Command(resume={"AMB-001": "n/a"}), thread)
    # submit a broken edit: the syntax gate must strip it and re-interrupt
    result = offline_graph.invoke(
        Command(resume={"action": "edited", "files": {"test_ts_001_budget.py": BROKEN_CODE}}),
        thread,
    )
    payload = result["__interrupt__"][0].value
    names = [f["file_name"] for f in payload["files"]]
    assert "test_ts_001_budget.py" not in names
    rejected = [r["file_name"] for r in payload["rejected_by_static_check"]]
    assert "test_ts_001_budget.py" in rejected


def test_review_gate_abort_skips_execution(offline_graph):
    thread = {"configurable": {"thread_id": "t4"}}
    result = offline_graph.invoke({}, thread)
    result = offline_graph.invoke(Command(resume={"AMB-001": "n/a"}), thread)
    result = offline_graph.invoke(Command(resume={"action": "abort"}), thread)
    assert result["review_action"] == "abort"
    assert "execution" not in result
    assert result["report_path"]
