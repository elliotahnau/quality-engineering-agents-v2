from qe_agent.cli import _exit_code
from qe_agent.schemas import Defect


def _defect(classification: str) -> Defect:
    return Defect(
        id="DEF-001",
        title="t",
        classification=classification,
        severity="medium",
        priority="P2",
        endpoint="GET /x",
        scenario_ids=[],
        test_ids=[],
        evidence="e",
        root_cause_hypothesis="r",
        suspected_owner="o",
    )


def test_exit_zero_when_no_real_defects():
    assert _exit_code({"defects": []}) == 0
    assert _exit_code({"defects": [_defect("flaky")]}) == 0


def test_exit_two_on_real_defects():
    assert _exit_code({"defects": [_defect("flaky"), _defect("real")]}) == 2


def test_exit_three_on_review_abort():
    assert _exit_code({"review_action": "abort", "defects": [_defect("real")]}) == 3
