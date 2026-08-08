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
