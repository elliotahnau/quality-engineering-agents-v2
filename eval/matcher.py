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
    require: tuple[str, ...] = ()  # all-of, for labels whose keywords are too generic alone


LABELS: list[Label] = [
    Label("BUG-001", "real", "POST /campaigns", ("daily_budget",)),
    Label("BUG-002", "real", "POST /campaigns", ("end_date", "start_date", "date range")),
    Label(
        # the defect must both name cpc AND claim the formula is wrong: a bare
        # "cpc" also appears when a zero-division defect enumerates the metric
        # fields, and a bare "inverted formula" appears in hypotheses about ctr
        "BUG-003",
        "real",
        "GET /campaigns/{id}/metrics",
        ("inverted", "formula", "incorrect", "wrong", "mismatch", "clicks/spend", "clicks / spend"),
        require=("cpc",),
    ),
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
        # about MISSING ROWS in the report — a bare "end_date" also appears in
        # unrelated /reports defects (e.g. validation-order claims)
        "BUG-006",
        "real",
        "POST /reports",
        ("row", "inclusive", "excludes", "omits", "off-by-one", "empty", "end date"),
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
    """The text keywords are matched against: title, root cause, and the
    structured repro.

    Quoted material is the enemy here. The evidence contract requires quoting
    the violated spec rule verbatim, and a quoted rule carries every keyword
    of its endpoint — the metrics rule contains "cpc" whether or not the
    defect is about CPC; a quoted request body names every creation field
    whichever one is at fault. Scanning those spans credited a formula bug no
    test had exercised. So:

    - evidence is scanned with quoted spans ('...' and "...") stripped: what
      remains is the model's own claim, which is exactly the signal.
    - `spec_refs` (pure spec quotes) and the repro payload (field names of
      the whole request) are excluded entirely.
    - the repro's method/path/status codes are included."""
    evidence = re.sub(r"'[^'\n]{0,400}'", " ", defect.evidence)
    evidence = re.sub(r'"[^"\n]{0,400}"', " ", evidence)
    parts = [defect.title, defect.root_cause_hypothesis, evidence]
    if defect.repro is not None:
        parts += [
            defect.repro.method,
            defect.repro.path,
            str(defect.repro.expected_status),
            str(defect.repro.observed_status),
        ]
    return " ".join(parts).lower()


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
        if label.require and not all(term in text for term in label.require):
            continue
        if any(keyword in text for keyword in label.keywords):
            matches.append(label)
    return matches
