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
        "BUG-008",
        "real",
        "PATCH /campaigns/{id}/budget",
        ("coerc", "bool", "true", "string", "non-numeric", "type"),
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


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


def split_endpoint(endpoint: str) -> tuple[str | None, str]:
    """'GET /campaigns/{campaign_id}/metrics' -> ('get', '/campaigns/{id}/metrics').

    The method is optional: models sometimes report the bare path, and losing
    a whole run's matches to that formatting choice would corrupt the metric
    rather than measure the pipeline.
    """
    text = re.sub(r"\s+", " ", endpoint.strip().lower()).rstrip("/")
    method = None
    head, _, rest = text.partition(" ")
    if head in HTTP_METHODS and rest:
        method, text = head, rest
    path = re.sub(r"\{[^}]*\}", "{id}", text)
    return method, re.sub(r"/<[^>]*>", "/{id}", path)


def normalize_endpoint(endpoint: str) -> str:
    """Path form used for comparison, method dropped."""
    return split_endpoint(endpoint)[1]


def defect_text(defect: Defect) -> str:
    return " ".join(
        [defect.title, defect.evidence, defect.root_cause_hypothesis, *defect.spec_refs]
    ).lower()


def match_defect(defect: Defect) -> list[Label]:
    """All labels this defect provides evidence for (possibly none).

    Paths must agree; the method is only compared when the defect states one,
    and the symptom keywords keep same-path labels apart.
    """
    method, path = split_endpoint(defect.endpoint)
    text = defect_text(defect)
    matches = []
    for label in LABELS:
        label_method, label_path = split_endpoint(label.endpoint)
        if label_path != path:
            continue
        if method is not None and label_method is not None and method != label_method:
            continue
        if any(keyword in text for keyword in label.keywords):
            matches.append(label)
    return matches
