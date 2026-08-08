"""Ad Operations Campaign API

Every planted bug is tagged with a `# BUG-xxx` / `# FLAKY-xxx` comment and
registered in bugs.yaml. Do not fix them; the QE agents are supposed to find
them, and the eval harness scores against bugs.yaml.

Flakiness is counter-based (every Nth call fails), not random, so eval runs
are reproducible. SUT_FLAKY_EVERY=0 disables it.
"""

import os
from datetime import date, timedelta
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# The OpenAPI metadata below (docstrings + Field descriptions) is the intended
# behavioral contract — the QE agents ground on GET /openapi.json only, never
# on this source file. Descriptions are uniform across all fields/endpoints so
# their presence doesn't hint at where bugs are. A few behaviors are left
# deliberately undocumented (name length, daily vs total budget relationship,
# report range inclusivity) to exercise the planner's ambiguity surfacing.
app = FastAPI(
    title="AdOps Campaign API",
    version="0.1.0",
    description=(
        "Internal ad-operations API for creating campaigns, controlling their "
        "lifecycle, and reading spend/performance data. Validation failures "
        "return HTTP 422; missing resources return HTTP 404."
    ),
)

CHANNELS = {"search", "display", "video", "social"}

# in-memory store; reset via reset_state() (used by tests)
_campaigns: dict[str, dict] = {}
_counters: dict[str, int] = {"resume": 0, "list": 0}


def reset_state() -> None:
    _campaigns.clear()
    _counters["resume"] = 0
    _counters["list"] = 0


def _flaky_every() -> int:
    """Read at call time so tests/eval can toggle without reloading the app."""
    raw = os.getenv("SUT_FLAKY_EVERY", "5")
    try:
        return int(raw)
    except ValueError:
        return 0


class CampaignCreate(BaseModel):
    name: str = Field(description="Campaign display name. Must be a non-empty string.")
    channel: str = Field(description="Ad channel. Must be one of: display, search, social, video.")
    total_budget: float = Field(
        description="Total campaign budget in account currency. Must be greater than 0."
    )
    daily_budget: float = Field(
        description="Daily spend cap in account currency. Must be greater than 0."
    )
    start_date: date = Field(description="Campaign start date (ISO 8601).")
    end_date: date = Field(
        description="Campaign end date (ISO 8601). Must be the same day as or later than start_date."
    )


class BudgetPatch(BaseModel):
    total_budget: float | None = Field(
        default=None, description="New total budget. If provided, must be greater than 0."
    )
    daily_budget: float | None = Field(
        default=None, description="New daily spend cap. If provided, must be greater than 0."
    )


class ReportRequest(BaseModel):
    campaign_id: str = Field(description="Id of the campaign to report on.")
    start_date: date = Field(description="Report range start date (ISO 8601).")
    end_date: date = Field(
        description="Report range end date (ISO 8601). Must not be earlier than start_date."
    )


@app.get("/health")
def health() -> dict:
    """Liveness check. Returns {"status": "ok"} when the service is up."""
    return {"status": "ok"}


@app.post("/campaigns", status_code=201)
def create_campaign(body: CampaignCreate) -> dict:
    """Create a campaign.

    Validation: name must be non-empty; channel must be a supported channel;
    total_budget must be greater than 0; daily_budget must be greater than 0;
    end_date must be the same day as or later than start_date. Any violation
    returns HTTP 422. On success returns HTTP 201 with the created campaign,
    including a server-generated id and status set to 'active'.
    """
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="name must not be empty")
    if body.channel not in CHANNELS:
        raise HTTPException(status_code=422, detail=f"channel must be one of {sorted(CHANNELS)}")
    if body.total_budget <= 0:
        raise HTTPException(status_code=422, detail="total_budget must be positive")
    # BUG-001: daily_budget is never validated — negative/zero values are accepted
    # even though the spec requires daily_budget > 0.
    # BUG-002: no cross-field validation — end_date earlier than start_date is accepted.
    campaign = {
        "id": str(uuid4()),
        "name": body.name,
        "channel": body.channel,
        "total_budget": body.total_budget,
        "daily_budget": body.daily_budget,
        "start_date": body.start_date.isoformat(),
        "end_date": body.end_date.isoformat(),
        "status": "active",
    }
    _campaigns[campaign["id"]] = campaign
    return campaign


@app.get("/campaigns")
def list_campaigns(channel: str | None = None) -> list[dict]:
    """List all campaigns. The optional channel query parameter filters the
    result to campaigns on that channel."""
    # FLAKY-002: every (N+2)th list call returns an EMPTY list with HTTP 200,
    # simulating a stale read from a lagging replica. Data-level flakiness —
    # the status code is fine, the content is wrong. Counter-based for
    # reproducibility.
    every = _flaky_every()
    if every > 0:
        _counters["list"] += 1
        if _counters["list"] % (every + 2) == 0:
            return []
    items = list(_campaigns.values())
    if channel is not None:
        items = [c for c in items if c["channel"] == channel]
    return items


@app.get("/campaigns/{campaign_id}")
def get_campaign(campaign_id: str) -> dict:
    """Fetch a single campaign by id. Returns HTTP 404 when no campaign with
    that id exists."""
    # BUG-007: unknown id raises KeyError -> 500, spec requires 404.
    return _campaigns[campaign_id]


@app.patch("/campaigns/{campaign_id}/budget")
def patch_budget(campaign_id: str, body: BudgetPatch) -> dict:
    """Update total_budget and/or daily_budget independently. The same
    positivity rules as creation apply (HTTP 422 on violation). Returns
    HTTP 404 for unknown campaign ids."""
    campaign = _campaigns.get(campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    if body.total_budget is not None:
        if body.total_budget <= 0:
            raise HTTPException(status_code=422, detail="total_budget must be positive")
        campaign["total_budget"] = body.total_budget
    if body.daily_budget is not None:
        if body.daily_budget <= 0:
            raise HTTPException(status_code=422, detail="daily_budget must be positive")
        campaign["daily_budget"] = body.daily_budget
    return campaign


@app.post("/campaigns/{campaign_id}/pause")
def pause_campaign(campaign_id: str) -> dict:
    """Pause a campaign: sets status to 'paused'. Pausing is idempotent —
    pausing an already-paused campaign leaves it paused. Returns HTTP 404
    for unknown campaign ids."""
    campaign = _campaigns.get(campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    # BUG-005: toggle instead of idempotent set — pausing a paused campaign
    # silently reactivates it. Spec requires pause to be idempotent.
    campaign["status"] = "paused" if campaign["status"] == "active" else "active"
    return campaign


@app.post("/campaigns/{campaign_id}/resume")
def resume_campaign(campaign_id: str) -> dict:
    """Resume a campaign: sets status to 'active'. Resuming is idempotent.
    Returns HTTP 404 for unknown campaign ids."""
    campaign = _campaigns.get(campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    # FLAKY-001: every Nth resume call fails with 503 ("scheduler busy"),
    # simulating a flaky downstream scheduler. Availability-level flakiness
    # on a bug-free endpoint, so its symptom never confounds a planted bug.
    every = _flaky_every()
    if every > 0:
        _counters["resume"] += 1
        if _counters["resume"] % every == 0:
            raise HTTPException(status_code=503, detail="campaign scheduler busy")
    campaign["status"] = "active"
    return campaign


@app.get("/campaigns/{campaign_id}/metrics")
def campaign_metrics(campaign_id: str) -> dict:
    """Performance metrics for a campaign: impressions, clicks, spend,
    revenue, ctr, cpc, roas.

    Formulas: ctr = clicks / impressions (reported as 0 when impressions is 0);
    cpc = spend / clicks (0 when clicks is 0); roas = revenue / spend (0 when
    spend is 0). A paused campaign reports zeroed volume metrics
    (impressions = clicks = 0, spend = revenue = 0) with HTTP 200.
    Returns HTTP 404 for unknown campaign ids."""
    campaign = _campaigns.get(campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="campaign not found")

    if campaign["status"] == "paused":
        impressions, clicks, spend, revenue = 0, 0, 0.0, 0.0
    else:
        seed = sum(ord(ch) for ch in campaign["id"])
        impressions = 1000 + (seed % 9000)
        clicks = impressions // 50
        spend = round(clicks * 0.4, 2)
        revenue = round(spend * 1.8, 2)

    # BUG-004: ZeroDivisionError -> 500 when impressions == 0 (paused campaign).
    # Spec requires ctr to be reported as 0 in that case.
    ctr = round(clicks / impressions, 4)
    # BUG-003: CPC formula inverted — spec is spend / clicks, this is clicks / spend.
    cpc = round(clicks / spend, 4) if spend else 0.0
    roas = round(revenue / spend, 4) if spend else 0.0
    return {
        "campaign_id": campaign_id,
        "impressions": impressions,
        "clicks": clicks,
        "spend": spend,
        "revenue": revenue,
        "ctr": ctr,
        "cpc": cpc,
        "roas": roas,
    }


@app.post("/reports")
def spend_report(body: ReportRequest) -> dict:
    """Daily spend report for a campaign between start_date and end_date:
    one row per day carrying that day's spend, plus a total_spend sum.
    Returns HTTP 422 when end_date is earlier than start_date, and HTTP 404
    for unknown campaign ids."""
    campaign = _campaigns.get(body.campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    if body.end_date < body.start_date:
        raise HTTPException(status_code=422, detail="end_date must be on or after start_date")
    # BUG-006: off-by-one — range excludes the end date although the spec
    # says the report covers both endpoints inclusively.
    n_days = (body.end_date - body.start_date).days
    rows = [
        {
            "date": (body.start_date + timedelta(days=i)).isoformat(),
            "spend": campaign["daily_budget"],
        }
        for i in range(n_days)
    ]
    return {
        "campaign_id": body.campaign_id,
        "rows": rows,
        "total_spend": round(sum(r["spend"] for r in rows), 2),
    }
