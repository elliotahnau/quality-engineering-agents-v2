from pathlib import Path

import pytest

from qe_agent.schemas import CaseResult, Defect, ExecutionReport
from qe_agent.stages.triage import (
    _enforce_flaky_rule,
    _enforce_known_owner,
    cluster_failures,
    parse_owners,
    resolve_owner,
)


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
    defect = _defect("real").model_copy(
        update={
            "scenario_ids": ["TS-001"],
            "test_ids": ["test_ts_001_a.py::test_ts_001_neg_budget"],
        }
    )
    defects = _enforce_flaky_rule([defect], clusters, execution)
    assert defects[0].classification == "real"


def test_flaky_rule_survives_reordered_llm_output():
    """Defects are paired with clusters by cited test id, so a model that
    answers out of order cannot make one cluster inherit another's verdict."""
    execution = make_execution()
    clusters = cluster_failures(execution)  # [TS-001 (persistent), TS-005 (recovered)]
    persistent = _defect("real").model_copy(
        update={
            "id": "DEF-002",
            "scenario_ids": ["TS-001"],
            "test_ids": ["test_ts_001_a.py::test_ts_001_neg_budget"],
        }
    )
    flaky_one = _defect("real")  # cites the TS-005 test, listed first
    adjusted = _enforce_flaky_rule([flaky_one, persistent], clusters, execution)
    assert [d.classification for d in adjusted] == ["flaky", "real"]


def test_flaky_rule_leaves_unmatched_defect_untouched():
    execution = make_execution()
    orphan = _defect("real").model_copy(update={"test_ids": ["test_unknown.py::test_x"]})
    assert (
        _enforce_flaky_rule([orphan], cluster_failures(execution), execution)[0].classification
        == "real"
    )


def test_owner_resolution_and_override():
    rules, default_owner = parse_owners(
        "owners:\n"
        "  - prefix: /campaigns/{id}/metrics\n"
        "    owner: analytics-team\n"
        "  - prefix: /campaigns\n"
        "    owner: campaign-core-team\n"
        "default_owner: platform-team\n"
    )
    # longest prefix wins over its parent path
    assert (
        resolve_owner("GET /campaigns/{campaign_id}/metrics", rules, default_owner)
        == "analytics-team"
    )
    assert resolve_owner("POST /campaigns", rules, default_owner) == "campaign-core-team"
    assert resolve_owner("GET /health", rules, default_owner) == "platform-team"

    invented = _defect("real").model_copy(update={"suspected_owner": "team-that-does-not-exist"})
    (fixed,) = _enforce_known_owner([invented], rules, default_owner)
    assert fixed.suspected_owner == "analytics-team"

    real_owner = _defect("real")  # already 'analytics-team'
    assert _enforce_known_owner([real_owner], rules, default_owner)[0].suspected_owner == (
        "analytics-team"
    )


@pytest.mark.parametrize(
    "bad_map",
    [
        "owners: []\n",  # no default_owner
        "owners:\n  - prefix: /x\ndefault_owner: t\n",  # entry without owner
        "owners:\n  - owner: t\ndefault_owner: t\n",  # entry without prefix
    ],
)
def test_malformed_ownership_map_is_rejected(bad_map):
    with pytest.raises(ValueError):
        parse_owners(bad_map)


def test_shipped_ownership_map_is_valid():
    rules, default_owner = parse_owners(Path("sut/owners.yaml").read_text())
    assert rules and default_owner
