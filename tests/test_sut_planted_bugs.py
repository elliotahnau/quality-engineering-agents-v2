import pytest
from fastapi.testclient import TestClient
from sut import app as sut_app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("SUT_FLAKY_EVERY", "0")
    sut_app.reset_state()
    with TestClient(sut_app.app, raise_server_exceptions=False) as c:
        yield c


def make_campaign(client, **overrides):
    body = {
        "name": "summer-sale",
        "channel": "search",
        "total_budget": 1000.0,
        "daily_budget": 100.0,
        "start_date": "2026-08-01",
        "end_date": "2026-08-10",
    }
    body.update(overrides)
    return client.post("/campaigns", json=body)


def test_happy_path_create_and_get(client):
    created = make_campaign(client)
    assert created.status_code == 201
    cid = created.json()["id"]
    assert client.get(f"/campaigns/{cid}").json()["status"] == "active"


def test_bug_001_negative_daily_budget_accepted(client):
    resp = make_campaign(client, daily_budget=-50.0)
    assert resp.status_code == 201  # spec requires 422


def test_bug_002_end_before_start_accepted(client):
    resp = make_campaign(client, start_date="2026-08-10", end_date="2026-08-01")
    assert resp.status_code == 201  # spec requires 422


def test_bug_003_cpc_formula_inverted(client):
    cid = make_campaign(client).json()["id"]
    m = client.get(f"/campaigns/{cid}/metrics").json()
    assert m["cpc"] == round(m["clicks"] / m["spend"], 4)  # spec: spend / clicks
    assert m["cpc"] != round(m["spend"] / m["clicks"], 4)


def test_bug_004_paused_metrics_500(client):
    cid = make_campaign(client).json()["id"]
    client.post(f"/campaigns/{cid}/pause")
    resp = client.get(f"/campaigns/{cid}/metrics")
    assert resp.status_code == 500  # spec requires 200 with ctr=0


def test_bug_005_pause_toggles(client):
    cid = make_campaign(client).json()["id"]
    client.post(f"/campaigns/{cid}/pause")
    second = client.post(f"/campaigns/{cid}/pause")
    assert second.json()["status"] == "active"  # spec: pause is idempotent


def test_bug_006_report_excludes_end_date(client):
    cid = make_campaign(client).json()["id"]
    resp = client.post(
        "/reports",
        json={"campaign_id": cid, "start_date": "2026-08-01", "end_date": "2026-08-03"},
    )
    rows = resp.json()["rows"]
    assert len(rows) == 2  # spec: 3 rows (inclusive)
    assert all(r["date"] != "2026-08-03" for r in rows)


def test_bug_007_unknown_campaign_500(client):
    resp = client.get("/campaigns/no-such-id")
    assert resp.status_code == 500  # spec requires 404


def test_bug_008_budget_type_coercion(client):
    cid = make_campaign(client).json()["id"]
    resp = client.patch(f"/campaigns/{cid}/budget", json={"total_budget": True})
    assert resp.status_code == 200  # spec requires 422
    assert resp.json()["total_budget"] == 1.0  # true silently became a budget
    resp = client.patch(f"/campaigns/{cid}/budget", json={"daily_budget": "50"})
    assert resp.status_code == 200
    assert resp.json()["daily_budget"] == 50.0


def test_flaky_001_resume_every_5th_call_503(client, monkeypatch):
    monkeypatch.setenv("SUT_FLAKY_EVERY", "5")
    sut_app.reset_state()
    cid = make_campaign(client).json()["id"]
    statuses = [client.post(f"/campaigns/{cid}/resume").status_code for _ in range(10)]
    assert statuses == [200] * 4 + [503] + [200] * 4 + [503]


def test_flaky_002_list_every_7th_call_stale_empty(client, monkeypatch):
    monkeypatch.setenv("SUT_FLAKY_EVERY", "5")
    sut_app.reset_state()
    make_campaign(client)
    responses = [client.get("/campaigns") for _ in range(7)]
    assert all(r.status_code == 200 for r in responses)  # data-level flakiness, not 5xx
    sizes = [len(r.json()) for r in responses]
    assert sizes == [1, 1, 1, 1, 1, 1, 0]  # 7th read is stale-empty


def test_metrics_not_flaky(client, monkeypatch):
    # planted-bug endpoints must stay deterministic so flaky symptoms never
    # confound BUG-003/BUG-004 detection
    monkeypatch.setenv("SUT_FLAKY_EVERY", "5")
    sut_app.reset_state()
    cid = make_campaign(client).json()["id"]
    statuses = {client.get(f"/campaigns/{cid}/metrics").status_code for _ in range(12)}
    assert statuses == {200}


def test_correct_behaviors_still_correct(client):
    # total_budget validation works
    assert make_campaign(client, total_budget=-1).status_code == 422
    # budget patch validates
    cid = make_campaign(client).json()["id"]
    assert client.patch(f"/campaigns/{cid}/budget", json={"daily_budget": -5}).status_code == 422
    # resume is a plain set
    client.post(f"/campaigns/{cid}/pause")
    assert client.post(f"/campaigns/{cid}/resume").json()["status"] == "active"


@pytest.fixture()
def clean_client(monkeypatch):
    """The eval's negative control: SUT_CLEAN=1 switches every planted bug off."""
    monkeypatch.setenv("SUT_FLAKY_EVERY", "0")
    monkeypatch.setenv("SUT_CLEAN", "1")
    sut_app.reset_state()
    with TestClient(sut_app.app, raise_server_exceptions=False) as c:
        yield c


def test_clean_mode_disables_every_planted_bug(clean_client):
    c = clean_client
    # BUG-001 / BUG-002: creation validates daily_budget and date order
    assert make_campaign(c, daily_budget=-50.0).status_code == 422
    assert make_campaign(c, start_date="2026-08-10", end_date="2026-08-01").status_code == 422
    cid = make_campaign(c).json()["id"]
    # BUG-003: cpc is spend / clicks
    m = c.get(f"/campaigns/{cid}/metrics").json()
    assert m["cpc"] == round(m["spend"] / m["clicks"], 4)
    # BUG-006: report range includes the end date
    report = c.post(
        "/reports",
        json={"campaign_id": cid, "start_date": "2026-08-01", "end_date": "2026-08-03"},
    )
    assert [r["date"] for r in report.json()["rows"]][-1] == "2026-08-03"
    assert len(report.json()["rows"]) == 3
    # BUG-004: paused metrics answer 200 with ctr=0
    c.post(f"/campaigns/{cid}/pause")
    resp = c.get(f"/campaigns/{cid}/metrics")
    assert resp.status_code == 200 and resp.json()["ctr"] == 0
    # BUG-005: pause is idempotent
    assert c.post(f"/campaigns/{cid}/pause").json()["status"] == "paused"
    # BUG-007: unknown id is 404
    assert c.get("/campaigns/no-such-id").status_code == 404
    # BUG-008: non-numeric budgets are rejected (both PATCH and POST)
    assert c.patch(f"/campaigns/{cid}/budget", json={"total_budget": True}).status_code == 422
    assert make_campaign(c, daily_budget="50").status_code == 422


def test_clean_mode_keeps_correct_behaviors(clean_client):
    assert make_campaign(clean_client, total_budget=-1).status_code == 422
    cid = make_campaign(clean_client).json()["id"]
    assert (
        clean_client.patch(f"/campaigns/{cid}/budget", json={"daily_budget": -5}).status_code == 422
    )
