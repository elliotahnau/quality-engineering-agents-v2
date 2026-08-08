"""Metric computation over repeated full-pipeline runs.

Metrics (all against sut/bugs.yaml labels, deterministic matching):
- detection rate: planted real bugs surfaced as defects (per run + pooled)
- classification accuracy: labeled-real defects classified "real",
  flaky-fault defects classified "flaky"
- false-positive rate: defects (classified real/flaky) matching no label.
  Unmatched defects classified "test_bug" are counted separately — that is
  the triager correctly blaming the generated test, not a false alarm.
- run-to-run variance is reported, not hidden: each metric is shown per run.
"""

import json
import uuid
from dataclasses import dataclass, field

from eval.matcher import LABELS, match_defect
from qe_agent.schemas import Defect

REAL_BUG_IDS = [label.id for label in LABELS if label.kind == "real"]
FLAKY_IDS = [label.id for label in LABELS if label.kind == "flaky"]


@dataclass
class RunScore:
    detected: set[str] = field(default_factory=set)  # label ids credited
    correct_classifications: int = 0
    matched_defects: int = 0
    false_positives: list[str] = field(default_factory=list)  # defect titles
    test_bug_noise: list[str] = field(default_factory=list)
    total_defects: int = 0

    @property
    def detection_rate(self) -> float:
        found = len(self.detected & set(REAL_BUG_IDS))
        return found / len(REAL_BUG_IDS)

    @property
    def classification_accuracy(self) -> float:
        return self.correct_classifications / self.matched_defects if self.matched_defects else 0.0

    @property
    def false_positive_rate(self) -> float:
        return len(self.false_positives) / self.total_defects if self.total_defects else 0.0


def score_run(defects: list[Defect]) -> RunScore:
    score = RunScore(total_defects=len(defects))
    for defect in defects:
        labels = match_defect(defect)
        if not labels:
            if defect.classification == "test_bug":
                score.test_bug_noise.append(defect.title)
            else:
                score.false_positives.append(defect.title)
            continue
        score.matched_defects += 1
        score.detected.update(label.id for label in labels)
        # a defect may credit several labels; classification is judged against
        # the strictest expectation: real wins over flaky when both match
        expected = "real" if any(lb.kind == "real" for lb in labels) else "flaky"
        if defect.classification == expected:
            score.correct_classifications += 1
    return score


def run_pipeline_once(graph) -> tuple[list[Defect], dict]:
    """One full --auto pipeline run against the already-running SUT."""
    thread = {"configurable": {"thread_id": uuid.uuid4().hex}}
    result = graph.invoke({"auto_approve": True}, thread)
    return list(result.get("defects") or []), result


def render_results(scores: list[RunScore]) -> str:
    lines = [
        "# Eval results — metrics over repeated pipeline runs",
        "",
        f"Runs: {len(scores)} | Ground truth: {len(REAL_BUG_IDS)} planted bugs, "
        f"{len(FLAKY_IDS)} flaky faults (sut/bugs.yaml)",
        "",
        "| run | detection rate | detected | classification acc | FP rate | "
        "false positives | test_bug noise |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(scores, 1):
        detected = ", ".join(sorted(s.detected)) or "-"
        lines.append(
            f"| {i} | {s.detection_rate:.0%} ({len(s.detected & set(REAL_BUG_IDS))}/"
            f"{len(REAL_BUG_IDS)}) | {detected} | {s.classification_accuracy:.0%} "
            f"({s.correct_classifications}/{s.matched_defects}) | {s.false_positive_rate:.0%} "
            f"({len(s.false_positives)}/{s.total_defects}) | "
            f"{'; '.join(s.false_positives) or '-'} | {'; '.join(s.test_bug_noise) or '-'} |"
        )
    pooled = set().union(*(s.detected for s in scores)) if scores else set()
    pooled_real = pooled & set(REAL_BUG_IDS)
    missed = sorted(set(REAL_BUG_IDS) - pooled_real)
    lines += [
        "",
        f"- pooled detection (found in >=1 run): {len(pooled_real)}/{len(REAL_BUG_IDS)}"
        f" — missed: {', '.join(missed) or 'none'}",
        f"- mean detection rate: {sum(s.detection_rate for s in scores) / len(scores):.0%}",
        f"- mean classification accuracy: "
        f"{sum(s.classification_accuracy for s in scores) / len(scores):.0%}",
        f"- mean false-positive rate: "
        f"{sum(s.false_positive_rate for s in scores) / len(scores):.0%}",
    ]
    return "\n".join(lines)


def dump_defects(defects: list[Defect]) -> str:
    return json.dumps([d.model_dump() for d in defects], indent=2)
