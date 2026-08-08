import pytest
from fastapi.testclient import TestClient
from sut.app import app


@pytest.fixture(scope="module")
def spec() -> dict:
    with TestClient(app) as client:
        return client.get("/openapi.json").json()


def test_every_operation_has_a_description(spec):
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            assert op.get("description"), f"{method.upper()} {path} has no description"


def test_every_request_field_has_a_description(spec):
    for name in ("CampaignCreate", "BudgetPatch", "ReportRequest"):
        schema = spec["components"]["schemas"][name]
        for field, prop in schema["properties"].items():
            assert prop.get("description"), f"{name}.{field} has no description"


def test_metric_formulas_are_documented(spec):
    desc = spec["paths"]["/campaigns/{campaign_id}/metrics"]["get"]["description"]
    assert "spend / clicks" in desc  # cpc — the planted bug inverts this
    assert "clicks / impressions" in desc
    assert "paused campaign reports zeroed" in desc


def test_deliberate_gaps_stay_undocumented(spec):
    # report range inclusivity is an ambiguity bait for the planner
    reports_desc = spec["paths"]["/reports"]["post"]["description"]
    assert "inclusive" not in reports_desc.lower()
    # name max length is another bait
    name_desc = spec["components"]["schemas"]["CampaignCreate"]["properties"]["name"]["description"]
    assert "length" not in name_desc.lower()
