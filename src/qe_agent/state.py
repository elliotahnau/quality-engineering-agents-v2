"""The LangGraph state shared across all stages."""

from typing import TypedDict

from qe_agent.schemas import (
    Defect,
    ExecutionReport,
    FailureCluster,
    GeneratedTest,
    SystemModel,
    TestPlan,
)


class QEState(TypedDict, total=False):
    # run controls
    auto_approve: bool
    run_dir: str
    # grounding (live OpenAPI spec only — SUT source code is never read)
    openapi_raw: str
    spec_spotlighted: str
    injection_warnings: list[str]
    system_model: SystemModel
    # planning
    plan: TestPlan
    human_answers: dict[str, str]
    # generation
    generated_tests: list[GeneratedTest]
    rejected_tests: list[dict]
    revision_feedback: dict[str, str]  # scenario_id -> reviewer feedback
    revision_round: int
    # review gate outcome: approve | revise | edited | abort
    review_action: str
    # execution
    execution: ExecutionReport
    # triage
    clusters: list[FailureCluster]
    defects: list[Defect]
    report_path: str
    defects_json_path: str
