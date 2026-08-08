"""Deterministic matching of reported defects to planted ground-truth labels.

No LLM judging (a deliberate scope decision): a defect matches a label when
its endpoint normalizes to the label's endpoint AND its text mentions any of
the label's symptom keywords. One defect may match several labels — the
pipeline legitimately merges e.g. two validation bugs on the same endpoint
into one defect, and both deserve detection credit.
"""

import re
from dataclasses import dataclass

from qe_agent.schemas import Defect


@dataclass(frozen=True)
class Label:
    id: str
    kind: str  # "real" | "flaky"
    endpoint: str
    keywords: tuple[str, ...]  # any-of, matched against lowercased defect text


LABELS: list[Label] = [
    Label("BUG-001", "real", "POST /campaigns", ("daily_budget",)),
    Label("BUG-002", "real", "POST /campaigns", ("end_date", "start_date", "date range")),
    Label("BUG-003", "real", "GET /campaigns/{id}/metrics", ("cpc",)),
    Label(
        "BUG-004",
        "real",
        "GET /campaigns/{id}/metrics",
        ("paused", "ctr", "impressions", "zerodivision", "zero division", "500"),
    ),
    Label(
        "BUG-005",
        "real",
        "POST /campaigns/{id}/pause",
        ("idempot", "toggle", "status", "reactivat"),
    ),
    Label(
        "BUG-006",
        "real",
        "POST /reports",
        ("end_date", "row", "inclusive", "excludes", "missing", "off-by-one"),
    ),
    Label(
        "BUG-007",
        "real",
        "GET /campaigns/{id}",
        ("404", "500", "unknown", "not found", "keyerror", "non-existent"),
    ),
    Label(
        "FLAKY-001",
        "flaky",
        "POST /campaigns/{id}/resume",
        ("503", "unavailable", "busy", "intermittent", "transient"),
    ),
    Label(
        "FLAKY-002",
        "flaky",
        "GET /campaigns",
        ("empty", "stale", "intermittent", "missing campaign"),
    ),
]


def normalize_endpoint(endpoint: str) -> str:
    """'GET /campaigns/{campaign_id}/metrics' -> 'get /campaigns/{id}/metrics'."""
    text = endpoint.strip().lower().rstrip("/")
    text = re.sub(r"\{[^}]*\}", "{id}", text)
    text = re.sub(r"/<[^>]*>", "/{id}", text)
    return re.sub(r"\s+", " ", text)


def defect_text(defect: Defect) -> str:
    return " ".join(
        [defect.title, defect.evidence, defect.root_cause_hypothesis, *defect.spec_refs]
    ).lower()


def match_defect(defect: Defect) -> list[Label]:
    """All labels this defect provides evidence for (possibly none)."""
    endpoint = normalize_endpoint(defect.endpoint)
    text = defect_text(defect)
    return [
        label
        for label in LABELS
        if normalize_endpoint(label.endpoint) == endpoint
        and any(keyword in text for keyword in label.keywords)
    ]
