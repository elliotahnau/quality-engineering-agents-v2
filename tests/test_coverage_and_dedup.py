"""Offline tests for the coverage accounting (scenario funnel, operation
coverage) and the deterministic file-name dedup. No LLM, no Docker."""

from qe_agent.report import operation_coverage, scenario_funnel
from qe_agent.schemas import (
    CaseResult,
    EndpointUnderstanding,
    ExecutionReport,
    GeneratedTest,
    RiskLevel,
    SystemModel,
    TestPlan,
    TestScenario,
    TestType,
)
from qe_agent.stages.generation import _dedupe_file_names


def _scenario(sid: str, endpoints: list[str]) -> TestScenario:
    return TestScenario(
        id=sid,
        title=f"scenario {sid}",
        basis="spec",
        risk=RiskLevel.medium,
        risk_rationale="r",
        test_types=[TestType.negative],
        steps=["s"],
        expected="e",
        endpoints=endpoints,
    )


def _plan(scenarios: list[TestScenario]) -> TestPlan:
    return TestPlan(
        summary="s",
        in_scope=[],
        out_of_scope=[],
        entry_criteria=[],
        exit_criteria=[],
        scenarios=scenarios,
    )


def _model(ops: list[tuple[str, str]]) -> SystemModel:
    return SystemModel(
        summary="s",
        entities=[],
        endpoints=[
            EndpointUnderstanding(method=m, path=p, purpose="", documented_rules=[]) for m, p in ops
        ],
        inferred_invariants=[],
        risk_areas=[],
        undocumented_aspects=[],
    )


def test_dedupe_file_names_disambiguates_collisions():
    tests = [
        GeneratedTest(scenario_id="TS-001", file_name="test_budget.py", code="a = 1"),
        GeneratedTest(scenario_id="TS-002", file_name="test_budget.py", code="b = 2"),
        GeneratedTest(scenario_id="TS-003", file_name="test_budget.py", code="c = 3"),
    ]
    deduped = _dedupe_file_names(tests)
    assert [t.file_name for t in deduped] == [
        "test_budget.py",
        "test_budget_2.py",
        "test_budget_3.py",
    ]
    # nothing dropped, order preserved
    assert [t.code for t in deduped] == ["a = 1", "b = 2", "c = 3"]


def test_dedupe_leaves_unique_names_alone():
    tests = [
        GeneratedTest(scenario_id="TS-001", file_name="test_a.py", code="x"),
        GeneratedTest(scenario_id="TS-002", file_name="test_b.py", code="y"),
    ]
    assert _dedupe_file_names(tests) == tests


def test_scenario_funnel_accounts_for_every_scenario():
    scenarios = [_scenario(f"TS-00{i}", []) for i in range(1, 6)]
    state = {
        "plan": _plan(scenarios),
        "generated_tests": [
            GeneratedTest(scenario_id="TS-001", file_name="test_a.py", code="x"),
            GeneratedTest(scenario_id="TS-004", file_name="test_d.py", code="x"),
        ],
        "rejected_tests": [
            {"scenario_id": "TS-002", "file_name": "test_b.py", "violations": ["syntax"]}
        ],
        "excluded_scenarios": ["TS-003"],
        "execution": ExecutionReport(
            retries=2,
            sandbox="docker",
            results=[CaseResult(test_id="test_a.py::t", scenario_id="TS-001", attempts=["passed"])],
        ),
    }
    statuses = {row["id"]: row["status"] for row in scenario_funnel(state)}
    assert statuses == {
        "TS-001": "executed",
        "TS-002": "rejected by static check",
        "TS-003": "excluded by reviewer",
        "TS-004": "generated, not executed",
        "TS-005": "not generated",
    }


def test_operation_coverage_counts_only_executed_scenarios():
    scenarios = [
        _scenario("TS-001", ["POST /campaigns"]),
        # dropped scenario: its operations must NOT earn coverage credit
        _scenario("TS-002", ["get /campaigns/{campaign_id}/metrics/"]),
    ]
    state = {
        "plan": _plan(scenarios),
        "system_model": _model(
            [
                ("POST", "/campaigns"),
                ("GET", "/campaigns/{id}/metrics"),
                ("POST", "/reports"),
            ]
        ),
        "generated_tests": [GeneratedTest(scenario_id="TS-001", file_name="test_a.py", code="x")],
        "execution": ExecutionReport(
            retries=2,
            sandbox="docker",
            results=[CaseResult(test_id="test_a.py::t", scenario_id="TS-001", attempts=["passed"])],
        ),
    }
    covered, uncovered = operation_coverage(state)
    assert covered == ["POST /campaigns"]
    # path-parameter names and trailing slashes are normalized away
    assert uncovered == ["GET /campaigns/{id}/metrics", "POST /reports"]
