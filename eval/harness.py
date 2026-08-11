"""Metric computation over repeated full-pipeline runs.

Metrics (against sut/bugs.yaml labels, deterministic matching; unmatched
defects are judged by replaying their structured repro against the live SUT):
- recall, label-level: planted real bugs and planted flaky faults surfaced
  as defects (per run + pooled)
- classification: a [expected kind x predicted classification] confusion
  matrix over matched defects, plus the derived accuracy
- precision, defect-level: share of filed real/flaky defects that point at
  genuine SUT behavior (label-matched or replay-reproduced)
- false positives, split three ways: replay-refuted (confirmed false alarm),
  replay-reproduced (genuine behavior missing from the label set — a ground
  truth gap, not an agent error), and not-replayable (unverified claim, the
  old hand-check bucket). Unmatched defects classified "test_bug" are counted
  separately — that is the triager correctly blaming the generated test.
- run-to-run variance is reported, not hidden: each metric is shown per run.

score_clean_run scores the negative control (--clean): the SUT has no planted
bugs, so labels are meaningless and replay is the only judge.
"""

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from eval.matcher import LABELS, match_defect
from eval.replay import NOT_REPLAYABLE, NOT_REPRODUCED, REPRODUCED
from qe_agent.schemas import Defect

REAL_BUG_IDS = [label.id for label in LABELS if label.kind == "real"]
FLAKY_IDS = [label.id for label in LABELS if label.kind == "flaky"]

CLASSIFICATIONS = ("real", "flaky", "test_bug")

Replay = Callable[[Defect], str]


@dataclass
class RunScore:
    detected: set[str] = field(default_factory=set)  # label ids credited
    # (expected kind, predicted classification) counts over matched defects
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    correct_classifications: int = 0
    matched_defects: int = 0
    false_positives: list[str] = field(default_factory=list)  # replay refuted
    unverified_claims: list[str] = field(default_factory=list)  # not replayable
    unlabeled_reproduced: list[str] = field(default_factory=list)  # label-set gap
    test_bug_noise: list[str] = field(default_factory=list)
    total_defects: int = 0

    @property
    def detection_rate(self) -> float:
        """Label-level recall over the planted real bugs."""
        return len(self.detected & set(REAL_BUG_IDS)) / len(REAL_BUG_IDS)

    @property
    def flaky_detection_rate(self) -> float:
        """Label-level recall over the planted flaky faults."""
        return len(self.detected & set(FLAKY_IDS)) / len(FLAKY_IDS)

    @property
    def classification_accuracy(self) -> float:
        return self.correct_classifications / self.matched_defects if self.matched_defects else 0.0

    @property
    def claim_precision(self) -> float:
        """Share of filed real/flaky defects that point at genuine behavior
        (label-matched or replay-reproduced)."""
        claims = self.total_defects - len(self.test_bug_noise)
        genuine = self.matched_defects + len(self.unlabeled_reproduced)
        return genuine / claims if claims else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Confirmed false alarms only (replay refuted the claim)."""
        return len(self.false_positives) / self.total_defects if self.total_defects else 0.0

    @property
    def false_positive_rate_upper(self) -> float:
        """Upper bound: confirmed false alarms plus unverifiable claims."""
        if not self.total_defects:
            return 0.0
        return (len(self.false_positives) + len(self.unverified_claims)) / self.total_defects


def _bucket_unmatched(defect: Defect, replay: Replay | None) -> str:
    """Which bucket an unlabeled real/flaky claim belongs in."""
    return replay(defect) if replay else NOT_REPLAYABLE


def score_run(defects: list[Defect], replay: Replay | None = None) -> RunScore:
    """Score one pipeline run. replay (optional) judges unmatched defects
    against the live SUT; without it they land in unverified_claims."""
    score = RunScore(total_defects=len(defects))
    for defect in defects:
        labels = match_defect(defect)
        if not labels:
            if defect.classification == "test_bug":
                score.test_bug_noise.append(defect.title)
                continue
            verdict = _bucket_unmatched(defect, replay)
            if verdict == REPRODUCED:
                score.unlabeled_reproduced.append(defect.title)
            elif verdict == NOT_REPRODUCED:
                score.false_positives.append(defect.title)
            else:
                score.unverified_claims.append(defect.title)
            continue
        score.matched_defects += 1
        score.detected.update(label.id for label in labels)
        # a defect may credit several labels; classification is judged against
        # the strictest expectation: real wins over flaky when both match
        expected = "real" if any(lb.kind == "real" for lb in labels) else "flaky"
        key = (expected, defect.classification)
        score.confusion[key] = score.confusion.get(key, 0) + 1
        if defect.classification == expected:
            score.correct_classifications += 1
    return score


@dataclass
class CleanRunScore:
    """Negative control: no planted bugs, so replay is the only judge."""

    false_alarms: list[str] = field(default_factory=list)  # replay refuted
    unverified_claims: list[str] = field(default_factory=list)  # not replayable
    reproduced: list[str] = field(default_factory=list)  # genuine unspecified behavior
    test_bug_noise: list[str] = field(default_factory=list)
    total_defects: int = 0


def score_clean_run(defects: list[Defect], replay: Replay | None = None) -> CleanRunScore:
    """Label matching is meaningless against a clean SUT (the bugs the labels
    describe are switched off), so every real/flaky claim goes through replay."""
    score = CleanRunScore(total_defects=len(defects))
    for defect in defects:
        if defect.classification == "test_bug":
            score.test_bug_noise.append(defect.title)
            continue
        verdict = _bucket_unmatched(defect, replay)
        if verdict == REPRODUCED:
            score.reproduced.append(defect.title)
        elif verdict == NOT_REPRODUCED:
            score.false_alarms.append(defect.title)
        else:
            score.unverified_claims.append(defect.title)
    return score


def run_pipeline_once(graph) -> tuple[list[Defect], dict]:
    """One full --auto pipeline run against the already-running SUT."""
    thread = {"configurable": {"thread_id": uuid.uuid4().hex}}
    result = graph.invoke({"auto_approve": True}, thread)
    return list(result.get("defects") or []), result


def _pooled_confusion(scores: list[RunScore]) -> dict[tuple[str, str], int]:
    pooled: dict[tuple[str, str], int] = {}
    for s in scores:
        for key, n in s.confusion.items():
            pooled[key] = pooled.get(key, 0) + n
    return pooled


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def render_results(scores: list[RunScore]) -> str:
    lines = [
        "# Eval results — metrics over repeated pipeline runs",
        "",
        f"Runs: {len(scores)} | Ground truth: {len(REAL_BUG_IDS)} planted bugs, "
        f"{len(FLAKY_IDS)} flaky faults (sut/bugs.yaml)",
        "",
        "| run | real recall | detected | flaky recall | classification acc | "
        "precision | confirmed FP | unverified | label-set gaps | test_bug noise |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(scores, 1):
        detected = ", ".join(sorted(s.detected)) or "-"
        lines.append(
            f"| {i} | {s.detection_rate:.0%} ({len(s.detected & set(REAL_BUG_IDS))}/"
            f"{len(REAL_BUG_IDS)}) | {detected} | {s.flaky_detection_rate:.0%} "
            f"({len(s.detected & set(FLAKY_IDS))}/{len(FLAKY_IDS)}) "
            f"| {s.classification_accuracy:.0%} ({s.correct_classifications}/"
            f"{s.matched_defects}) | {s.claim_precision:.0%} "
            f"| {'; '.join(s.false_positives) or '-'} "
            f"| {'; '.join(s.unverified_claims) or '-'} "
            f"| {'; '.join(s.unlabeled_reproduced) or '-'} "
            f"| {'; '.join(s.test_bug_noise) or '-'} |"
        )

    pooled = set().union(*(s.detected for s in scores)) if scores else set()
    pooled_real = pooled & set(REAL_BUG_IDS)
    pooled_flaky = pooled & set(FLAKY_IDS)
    missed = sorted((set(REAL_BUG_IDS) | set(FLAKY_IDS)) - pooled)
    lines += [
        "",
        f"- pooled recall (found in >=1 run): real {len(pooled_real)}/{len(REAL_BUG_IDS)}, "
        f"flaky {len(pooled_flaky)}/{len(FLAKY_IDS)}"
        f" — missed: {', '.join(missed) or 'none'}",
        f"- mean real recall: {_mean([s.detection_rate for s in scores]):.0%} | "
        f"mean flaky recall: {_mean([s.flaky_detection_rate for s in scores]):.0%}",
        f"- mean classification accuracy: "
        f"{_mean([s.classification_accuracy for s in scores]):.0%}",
        f"- mean defect precision: {_mean([s.claim_precision for s in scores]):.0%} "
        "(share of filed real/flaky defects pointing at genuine behavior)",
        f"- mean confirmed-FP rate: {_mean([s.false_positive_rate for s in scores]):.0%} "
        f"(upper bound incl. unverified: "
        f"{_mean([s.false_positive_rate_upper for s in scores]):.0%})",
    ]

    confusion = _pooled_confusion(scores)
    if confusion:
        lines += [
            "",
            "## Pooled classification confusion matrix (matched defects)",
            "",
            "| expected \\ predicted | real | flaky | test_bug |",
            "|---|---|---|---|",
        ]
        for expected in ("real", "flaky"):
            row = " | ".join(str(confusion.get((expected, p), 0)) for p in CLASSIFICATIONS)
            lines.append(f"| {expected} | {row} |")
    return "\n".join(lines)


def render_clean_results(scores: list[CleanRunScore]) -> str:
    lines = [
        "# Eval results — negative control (clean SUT, no planted bugs)",
        "",
        "Every real/flaky defect filed against the clean SUT is judged by "
        "replaying its repro: refuted -> confirmed false alarm; reproduced -> "
        "genuine behavior the spec leaves unspecified (not an agent error); "
        "no repro -> unverified claim.",
        "",
        "| run | defects filed | confirmed false alarms | unverified | "
        "reproduced (spec gaps) | test_bug noise |",
        "|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(scores, 1):
        lines.append(
            f"| {i} | {s.total_defects} "
            f"| {'; '.join(s.false_alarms) or '-'} "
            f"| {'; '.join(s.unverified_claims) or '-'} "
            f"| {'; '.join(s.reproduced) or '-'} "
            f"| {'; '.join(s.test_bug_noise) or '-'} |"
        )
    total = sum(s.total_defects for s in scores)
    alarms = sum(len(s.false_alarms) + len(s.unverified_claims) for s in scores)
    lines += [
        "",
        f"- false-alarm budget: {alarms}/{total} defects were confirmed or "
        "potential false alarms across all clean runs (0 is the target)",
    ]
    return "\n".join(lines)


def dump_defects(defects: list[Defect]) -> str:
    return json.dumps([d.model_dump() for d in defects], indent=2)
