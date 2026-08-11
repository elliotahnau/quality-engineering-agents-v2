"""Replay a defect's structured repro against a live SUT.

An unmatched defect (no bugs.yaml label) is only a false positive if the
behavior it claims does NOT reproduce. That judgment used to be a hand check;
for any defect carrying a single-request repro the harness now makes it
automatically, by sending the literal request and comparing status codes.
"""

import re

from qe_agent.schemas import Defect

REPRODUCED = "reproduced"
NOT_REPRODUCED = "not_reproduced"
NOT_REPLAYABLE = "not_replayable"

# valid campaign used to satisfy an {id} placeholder in a repro path
_FRESH_CAMPAIGN = {
    "name": "replay-probe",
    "channel": "search",
    "total_budget": 1000.0,
    "daily_budget": 100.0,
    "start_date": "2026-01-01",
    "end_date": "2026-01-10",
}


def replay_defect(defect: Defect, client) -> str:
    """Verdict on one defect against a live SUT.

    client is httpx.Client-compatible (``request(method, url, json=...)``),
    so tests can pass a FastAPI TestClient.

    - reproduced:     the SUT answered with the defect's observed status again
                      (and not the spec-expected one) — the claim stands.
    - not_reproduced: the SUT answered differently — a confirmed false alarm.
    - not_replayable: no structured repro, or it needs state a single fresh
                      campaign cannot satisfy.
    """
    repro = defect.repro
    if repro is None:
        return NOT_REPLAYABLE
    path = repro.path
    if "{" in path:
        # one resource placeholder is the common case; satisfy it with a
        # fresh campaign so ids from the original run don't matter
        created = client.request("POST", "/campaigns", json=_FRESH_CAMPAIGN)
        if created.status_code != 201:
            return NOT_REPLAYABLE
        path = re.sub(r"\{[^}]*\}", created.json()["id"], path)
    try:
        resp = client.request(repro.method.upper(), path, json=repro.payload)
    except Exception:
        return NOT_REPLAYABLE
    if resp.status_code == repro.observed_status and resp.status_code != repro.expected_status:
        return REPRODUCED
    return NOT_REPRODUCED
