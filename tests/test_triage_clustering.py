from qe_agent.schemas import CaseResult, Defect, ExecutionReport
from qe_agent.stages.triage import _enforce_flaky_rule, cluster_failures


def make_execution() -> ExecutionReport:
    return ExecutionReport(
        retries=2,
        sandbox="subprocess",
        results=[
            # deterministic failure, same signature twice -> one cluster
            CaseResult(
                test_id="test_ts_001_a.py::test_ts_001_neg_budget",
                scenario_id="TS-001",
                attempts=["failed", "failed", "failed"],
                message="assert 201 == 422",
            ),
            CaseResult(
                test_id="test_ts_001_a.py::test_ts_001_zero_budget",
                scenario_id="TS-001",
                attempts=["failed", "failed", "failed"],
                message="assert 201 == 422",
            ),
            # flaky: failed then recovered
            CaseResult(
                test_id="test_ts_005_m.py::test_ts_005_metrics",
                scenario_id="TS-005",
                attempts=["failed", "passed"],
                message="assert 503 == 200",
            ),
            # passing test -> not clustered
            CaseResult(
                test_id="test_ts_002_ok.py::test_ts_002_happy",
                scenario_id="TS-002",
                attempts=["passed"],
            ),
        ],
    )


def test_clustering_groups_by_signature():
    clusters = cluster_failures(make_execution())
    assert len(clusters) == 2
    by_scenario = {c.scenario_ids[0]: c for c in clusters}
    assert len(by_scenario["TS-001"].test_ids) == 2
    assert "ALL tests passed on retry" in by_scenario["TS-005"].flaky_evidence
    assert "Still failing" in by_scenario["TS-001"].flaky_evidence


def _defect(classification: str) -> Defect:
    return Defect(
        id="DEF-001",
        title="x",
        classification=classification,
        severity="medium",
        priority="P2",
        endpoint="GET /campaigns/{id}/metrics",
        scenario_ids=["TS-005"],
        test_ids=["test_ts_005_m.py::test_ts_005_metrics"],
        evidence="e",
        root_cause_hypothesis="r",
        suspected_owner="analytics-team",
    )


def test_flaky_rule_overrides_llm():
    execution = make_execution()
    clusters = [c for c in cluster_failures(execution) if c.scenario_ids == ["TS-005"]]
    defects = _enforce_flaky_rule([_defect("real")], clusters, execution)
    assert defects[0].classification == "flaky"


def test_flaky_rule_keeps_real_for_persistent_failures():
    execution = make_execution()
    clusters = [c for c in cluster_failures(execution) if c.scenario_ids == ["TS-001"]]
    defect = _defect("real").model_copy(update={"scenario_ids": ["TS-001"]})
    defects = _enforce_flaky_rule([defect], clusters, execution)
    assert defects[0].classification == "real"
