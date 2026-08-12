"""Offline tests for the eval harness: matching, scoring, replay, and
poisoned-spec construction. No LLM, no Docker."""

from pathlib import Path

import pytest
import yaml
from eval.harness import REAL_BUG_IDS, score_clean_run, score_run
from eval.injection import CANARY, RESPONSE_DIRECTIVE, VARIANTS
from eval.matcher import LABELS, match_defect, normalize_endpoint
from eval.replay import NOT_REPLAYABLE, NOT_REPRODUCED, REPRODUCED, replay_defect
from fastapi.testclient import TestClient
from sut import app as sut_app

from qe_agent.schemas import Defect, Repro
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


def test_endpoint_without_a_method_still_matches():
    """Models sometimes report the bare path. Losing a whole run's matches to
    that formatting choice would corrupt the metric instead of measuring the
    pipeline, so the method is compared only when the defect states one."""
    bare = make_defect(
        endpoint="/campaigns/{campaign_id}/metrics",
        title="metrics 500 for paused campaign",
        evidence="expected 200 with ctr 0, observed 500",
    )
    assert "BUG-004" in [lb.id for lb in match_defect(bare)]


def test_stated_method_still_disambiguates_same_path():
    """GET /campaigns (flaky) and POST /campaigns (planted bugs) share a path."""
    listing = make_defect(
        endpoint="GET /campaigns",
        title="campaign list intermittently empty",
        evidence="list returned empty despite existing campaigns; passed on retry",
        classification="flaky",
    )
    assert [lb.id for lb in match_defect(listing)] == ["FLAKY-002"]


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
        title="campaign creation accepts non-positive daily_budget and end_date before start_date",
        evidence="see repro",
        root_cause_hypothesis="missing validation on daily_budget and on the date range",
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
        repro=Repro(
            method="GET", path="/campaigns/{id}/metrics", expected_status=200, observed_status=500
        ),
    )
    assert [lb.id for lb in match_defect(cpc)] == ["BUG-003"]
    assert "BUG-004" in [lb.id for lb in match_defect(paused)]


def test_spec_quote_in_evidence_does_not_credit_another_bug():
    """Regression for a real over-credit: the evidence contract quotes the
    violated spec rule verbatim, and the metrics rule contains 'cpc' — so a
    paused-500 defect scored a BUG-003 credit although no test had exercised
    the CPC formula. Keywords must never be matched against evidence or
    spec_refs, both of which embed spec quotes by design."""
    paused = make_defect(
        endpoint="GET /campaigns/{campaign_id}/metrics",
        title="metrics return 500 when volume is zero",
        evidence=(
            "GET metrics returned 500, expected 200 per spec rule: 'ctr = clicks / "
            "impressions (reported as 0 when impressions is 0); cpc = spend / clicks "
            "(0 when clicks is 0); roas = revenue / spend'"
        ),
        spec_refs=["GET metrics: 'cpc = spend / clicks (0 when clicks is 0)'"],
        root_cause_hypothesis="unhandled zero division for paused campaigns",
        repro=Repro(
            method="GET", path="/campaigns/{id}/metrics", expected_status=200, observed_status=500
        ),
    )
    matched = {lb.id for lb in match_defect(paused)}
    assert "BUG-003" not in matched
    assert "BUG-004" in matched


def test_score_run_metrics():
    defects = [
        make_defect(  # matches BUG-001, correctly real
            title="negative daily_budget accepted",
            evidence="daily_budget -1 accepted, expected 422",
        ),
        make_defect(  # flaky fault reported as real -> classification miss
            id="DEF-002",
            endpoint="POST /campaigns/{id}/resume",
            title="intermittent 503 on resume",
            evidence="503 scheduler busy, passed on retry",
            classification="real",
        ),
        make_defect(  # unmatched + no replay available -> unverified claim
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
    # without a replay judge, an unmatched claim stays an unverified claim
    assert score.false_positives == []
    assert score.unverified_claims == ["health check content-type"]
    assert score.test_bug_noise == ["generated test asserted wrong shape"]
    assert score.false_positive_rate == 0.0
    assert score.false_positive_rate_upper == 0.25
    # recall split by label kind, confusion over matched defects
    assert score.detection_rate == 1 / len(REAL_BUG_IDS)
    assert score.flaky_detection_rate == 0.5
    assert score.confusion == {("real", "real"): 1, ("flaky", "real"): 1}
    # 2 of 3 real/flaky claims point at genuine (labeled) behavior
    assert score.claim_precision == pytest.approx(2 / 3)


def test_score_run_replay_splits_unmatched_claims():
    """The replay verdict decides whether an unmatched claim is a confirmed
    false alarm, a label-set gap, or stays unverified."""
    reproduced = make_defect(
        # the deliberate spec gap: daily_budget > total_budget is not rejected
        id="DEF-010",
        endpoint="PATCH /campaigns/{id}/budget",
        title="daily exceeding total budget accepted",
    )
    refuted = make_defect(id="DEF-011", endpoint="GET /health", title="phantom 500")
    unverifiable = make_defect(id="DEF-012", endpoint="GET /health", title="vague claim")
    verdicts = {
        "DEF-010": REPRODUCED,
        "DEF-011": NOT_REPRODUCED,
        "DEF-012": NOT_REPLAYABLE,
    }
    score = score_run([reproduced, refuted, unverifiable], replay=lambda d: verdicts[d.id])
    assert score.unlabeled_reproduced == ["daily exceeding total budget accepted"]
    assert score.false_positives == ["phantom 500"]
    assert score.unverified_claims == ["vague claim"]
    assert score.false_positive_rate == pytest.approx(1 / 3)
    assert score.claim_precision == pytest.approx(1 / 3)


def test_score_clean_run_uses_replay_not_labels():
    """Against the clean SUT the labels describe switched-off bugs, so a
    label-matching defect must NOT earn detection credit — replay decides."""
    looks_like_bug_001 = make_defect(
        title="negative daily_budget accepted",
        evidence="daily_budget -1 accepted, expected 422",
    )
    genuine_gap = make_defect(id="DEF-002", title="patch coerces bool to budget")
    noise = make_defect(id="DEF-003", title="test asserted wrong shape", classification="test_bug")
    verdicts = {"DEF-001": NOT_REPRODUCED, "DEF-002": REPRODUCED}
    score = score_clean_run(
        [looks_like_bug_001, genuine_gap, noise], replay=lambda d: verdicts[d.id]
    )
    assert score.false_alarms == [looks_like_bug_001.title]
    assert score.reproduced == ["patch coerces bool to budget"]
    assert score.test_bug_noise == ["test asserted wrong shape"]


@pytest.fixture()
def sut_client(monkeypatch):
    monkeypatch.setenv("SUT_FLAKY_EVERY", "0")
    sut_app.reset_state()
    with TestClient(sut_app.app, raise_server_exceptions=False) as c:
        yield c


def test_replay_reproduces_planted_bug(sut_client):
    defect = make_defect(
        repro=Repro(
            method="POST",
            path="/campaigns",
            payload={
                "name": "x",
                "channel": "search",
                "total_budget": 100.0,
                "daily_budget": -50.0,
                "start_date": "2026-01-01",
                "end_date": "2026-01-10",
            },
            expected_status=422,
            observed_status=201,
        )
    )
    assert replay_defect(defect, sut_client) == REPRODUCED


def test_replay_refutes_false_claim_and_fills_placeholder(sut_client):
    # PATCH budget DOES validate positivity — this claim must be refuted;
    # the {id} placeholder is satisfied with a fresh campaign
    defect = make_defect(
        repro=Repro(
            method="PATCH",
            path="/campaigns/{id}/budget",
            payload={"daily_budget": -5},
            expected_status=422,
            observed_status=200,
        )
    )
    assert replay_defect(defect, sut_client) == NOT_REPRODUCED


def test_replay_without_structured_repro_is_not_replayable(sut_client):
    assert replay_defect(make_defect(), sut_client) == NOT_REPLAYABLE


def test_replay_against_clean_sut_refutes_planted_bug_claim(sut_client, monkeypatch):
    monkeypatch.setenv("SUT_CLEAN", "1")
    defect = make_defect(
        repro=Repro(
            method="POST",
            path="/campaigns",
            payload={
                "name": "x",
                "channel": "search",
                "total_budget": 100.0,
                "daily_budget": -50.0,
                "start_date": "2026-01-01",
                "end_date": "2026-01-10",
            },
            expected_status=422,
            observed_status=201,
        )
    )
    assert replay_defect(defect, sut_client) == NOT_REPRODUCED


def test_bug_008_type_coercion_matches():
    """The gap the pipeline itself found, promoted to ground truth."""
    defect = make_defect(
        endpoint="PATCH /campaigns/{campaign_id}/budget",
        title="budget accepts boolean via lax type coercion",
        evidence='{"total_budget": true} returned 200 with total_budget=1.0, expected 422',
    )
    assert [lb.id for lb in match_defect(defect)] == ["BUG-008"]


def test_repro_payload_field_names_do_not_cross_credit_sibling_bugs():
    """A creation payload names every field (daily_budget, start_date, ...)
    whichever one is at fault, so the payload is excluded from keyword text —
    otherwise any POST /campaigns defect with a repro credits BUG-001 and
    BUG-002 at once."""
    date_bug = make_defect(
        title="campaign accepts end_date earlier than start_date",
        root_cause_hypothesis="missing cross-field date validation",
        repro=Repro(
            method="POST",
            path="/campaigns",
            payload={
                "name": "x",
                "channel": "search",
                "total_budget": 100.0,
                "daily_budget": 10.0,
                "start_date": "2026-01-10",
                "end_date": "2026-01-01",
            },
            expected_status=422,
            observed_status=201,
        ),
    )
    assert [lb.id for lb in match_defect(date_bug)] == ["BUG-002"]


def test_duplicate_filings_are_measured():
    """Two defects crediting the same label = duplication a human lead would
    have merged; counted, not hidden."""
    first = make_defect(
        title="negative daily_budget accepted",
        evidence="daily_budget -1 accepted, expected 422",
    )
    second = make_defect(
        id="DEF-002",
        title="zero daily_budget accepted",
        evidence="daily_budget 0 accepted, expected 422",
    )
    score = score_run([first, second])
    assert score.label_credits == {"BUG-001": 2}
    assert score.duplicate_filings == 1


def test_response_injection_directive_is_scanner_visible():
    """The poisoned-response probe's directive must trip the pattern scanner,
    so the attack shows up in warnings even when the triager ignores it."""
    warnings = scan_artifact(RESPONSE_DIRECTIVE)
    assert any("result-tampering" in w for w in warnings)
    assert any("echo-injection" in w for w in warnings)


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
