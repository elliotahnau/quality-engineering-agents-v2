"""Offline tests for the eval harness: matching, scoring, and poisoned-spec
construction. No LLM, no Docker."""

from pathlib import Path

import yaml
from eval.harness import score_run
from eval.injection import CANARY, VARIANTS
from eval.matcher import LABELS, match_defect, normalize_endpoint

from qe_agent.schemas import Defect
from qe_agent.security import scan_artifact


def make_defect(**overrides) -> Defect:
    base = dict(
        id="DEF-001",
        title="t",
        classification="real",
        severity="high",
        priority="P1",
        endpoint="POST /campaigns",
        scenario_ids=["TS-001"],
        test_ids=["test_x.py::test_y"],
        evidence="e",
        root_cause_hypothesis="r",
        suspected_owner="o",
        spec_refs=[],
    )
    base.update(overrides)
    return Defect(**base)


def test_normalize_endpoint_unifies_path_params():
    assert normalize_endpoint("GET /campaigns/{campaign_id}/metrics") == normalize_endpoint(
        "get /campaigns/{id}/metrics/"
    )


def test_labels_stay_in_sync_with_bugs_yaml():
    manifest = yaml.safe_load(Path("sut/bugs.yaml").read_text())
    truth = {
        entry["id"]: normalize_endpoint(entry["endpoint"])
        for entry in manifest["bugs"] + manifest["flaky"]
    }
    ours = {label.id: normalize_endpoint(label.endpoint) for label in LABELS}
    assert ours == truth


def test_merged_validation_defect_credits_both_bugs():
    defect = make_defect(
        title="campaign creation accepts invalid input",
        evidence="daily_budget <= 0 accepted; end_date before start_date accepted (expected 422)",
    )
    assert {lb.id for lb in match_defect(defect)} == {"BUG-001", "BUG-002"}


def test_cpc_and_paused_metrics_are_distinguished():
    cpc = make_defect(
        endpoint="GET /campaigns/{campaign_id}/metrics",
        title="CPC formula inverted",
        evidence="cpc returned clicks/spend",
    )
    paused = make_defect(
        endpoint="GET /campaigns/{campaign_id}/metrics",
        title="metrics 500 for paused campaign",
        evidence="expected 200 with ctr 0, observed 500",
    )
    assert [lb.id for lb in match_defect(cpc)] == ["BUG-003"]
    assert "BUG-004" in [lb.id for lb in match_defect(paused)]


def test_score_run_metrics():
    defects = [
        make_defect(  # matches BUG-001, correctly real
            evidence="daily_budget -1 accepted, expected 422"
        ),
        make_defect(  # flaky fault reported as real -> classification miss
            id="DEF-002",
            endpoint="POST /campaigns/{id}/resume",
            title="intermittent 503 on resume",
            evidence="503 scheduler busy, passed on retry",
            classification="real",
        ),
        make_defect(  # unmatched + classified real -> false positive
            id="DEF-003",
            endpoint="GET /health",
            title="health check content-type",
            evidence="nitpick",
        ),
        make_defect(  # unmatched but test_bug -> counted as noise, not FP
            id="DEF-004",
            endpoint="GET /health",
            title="generated test asserted wrong shape",
            classification="test_bug",
        ),
    ]
    score = score_run(defects)
    assert score.detected == {"BUG-001", "FLAKY-001"}
    assert score.matched_defects == 2 and score.correct_classifications == 1
    assert score.false_positives == ["health check content-type"]
    assert score.test_bug_noise == ["generated test asserted wrong shape"]
    assert score.false_positive_rate == 0.25


def test_variants_poison_spec_and_scanner_catches_most():
    base = {
        "info": {"description": "Campaign API."},
        "paths": {"/campaigns": {"post": {"description": "Create."}}},
    }
    flagged = {}
    for variant in VARIANTS:
        poisoned = variant.build(base)
        assert poisoned != base  # directive actually lands in the spec
        assert base["info"]["description"] == "Campaign API."  # input not mutated
        import json

        flagged[variant.name] = bool(scan_artifact(json.dumps(poisoned)))
    assert all(flagged.values()), f"scanner missed: {[k for k, v in flagged.items() if not v]}"
    assert CANARY in str(VARIANTS[2].build(base))
