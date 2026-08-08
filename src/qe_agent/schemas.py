"""
These models are both the LangGraph state payloads and the structured-output
schemas handed to the LLM, so field descriptions works as prompt guidance.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"


class TestType(str, Enum):
    happy_path = "happy_path"
    boundary = "boundary"
    negative = "negative"
    error_handling = "error_handling"
    state_transition = "state_transition"
    idempotency = "idempotency"
    concurrency = "concurrency"


class EndpointUnderstanding(BaseModel):
    """The agent's grounded understanding of one API operation."""

    method: str
    path: str
    purpose: str = Field(description="What this operation does for the business")
    documented_rules: list[str] = Field(
        description="Behavioral rules stated in the spec (validation, status codes, formulas)"
    )


class SystemModel(BaseModel):
    """Output of the grounding stage: what the agent understood about the SUT."""

    summary: str = Field(description="What the system is and does, in a few sentences")
    entities: list[str] = Field(description="Core data entities and their key fields")
    endpoints: list[EndpointUnderstanding]
    inferred_invariants: list[str] = Field(
        description=(
            "Rules NOT stated in the spec but implied by domain knowledge or HTTP "
            "convention (e.g. standard metric formulas, idempotent lifecycle verbs, "
            "404 for missing resources, 5xx never acceptable). Mark each as INFERRED."
        )
    )
    risk_areas: list[str] = Field(
        description="Where failures would hurt most: money, state, calculations"
    )
    undocumented_aspects: list[str] = Field(
        description="Behaviors the spec leaves unspecified — candidates for ambiguities"
    )


class Ambiguity(BaseModel):
    """A spec gap the planner refuses to silently guess about."""

    id: str = Field(description="Stable id like AMB-001")
    question: str = Field(description="The question a QE engineer would ask the PM")
    context: str = Field(description="Which endpoint / spec section triggered this")
    assumption: str = Field(description="The default assumption used if the human does not answer")


class TestScenario(BaseModel):
    id: str = Field(description="Stable id like TS-001")
    title: str
    basis: str = Field(
        description=(
            "What this scenario verifies against: a documented spec rule "
            "(quote it, e.g. \"POST /campaigns: 'daily_budget must be greater than 0'\") "
            "or an inferred invariant (prefix with INFERRED:)"
        )
    )
    risk: RiskLevel
    risk_rationale: str = Field(description="Why this risk level: impact x likelihood")
    test_types: list[TestType]
    steps: list[str] = Field(description="Concrete API-level steps")
    expected: str = Field(description="Expected observable outcome (status codes, payload)")


class TestPlan(BaseModel):
    summary: str
    in_scope: list[str]
    out_of_scope: list[str]
    entry_criteria: list[str]
    exit_criteria: list[str]
    scenarios: list[TestScenario]
    ambiguities: list[Ambiguity] = Field(
        default_factory=list,
        description="Spec gaps to surface to a human instead of guessing",
    )


class GeneratedTest(BaseModel):
    scenario_id: str
    file_name: str = Field(description="e.g. test_ts_001_budget_validation.py")
    code: str = Field(description="Complete pytest module source")


class GeneratedTestBatch(BaseModel):
    tests: list[GeneratedTest]


class CaseResult(BaseModel):
    """Outcome of one test function across all attempts (first run + retries)."""

    test_id: str = Field(description="pytest nodeid, e.g. test_x.py::test_y")
    scenario_id: str | None = None
    attempts: list[Literal["passed", "failed", "error", "skipped"]]
    message: str = ""
    traceback: str = ""

    @property
    def final_outcome(self) -> str:
        return self.attempts[-1] if self.attempts else "skipped"

    @property
    def ever_passed(self) -> bool:
        return "passed" in self.attempts

    @property
    def ever_failed(self) -> bool:
        return any(a in ("failed", "error") for a in self.attempts)


class ExecutionReport(BaseModel):
    results: list[CaseResult]
    retries: int
    sandbox: str = Field(description="Sandbox backend used, e.g. 'subprocess'")
    duration_seconds: float = 0.0

    @property
    def failed_cases(self) -> list[CaseResult]:
        return [r for r in self.results if r.ever_failed]


class FailureCluster(BaseModel):
    """Failures grouped by signature before triage; one cluster -> one defect."""

    signature: str
    test_ids: list[str]
    scenario_ids: list[str]
    sample_message: str
    sample_traceback: str
    flaky_evidence: str = Field(
        description="Retry statistics, e.g. '2/3 attempts failed, passed on retry'"
    )


class Defect(BaseModel):
    id: str = Field(description="Stable id like DEF-001")
    title: str
    classification: Literal["real", "flaky", "test_bug"] = Field(
        description=(
            "real: SUT violates the spec deterministically; "
            "flaky: intermittent, passes on retry; "
            "test_bug: the generated test itself is wrong"
        )
    )
    severity: Literal["critical", "high", "medium", "low"]
    priority: Literal["P1", "P2", "P3"]
    endpoint: str = Field(description="Affected endpoint, e.g. 'POST /campaigns'")
    scenario_ids: list[str]
    test_ids: list[str]
    evidence: str = Field(description="Observed vs expected, with repro steps")
    root_cause_hypothesis: str
    suspected_owner: str
    spec_refs: list[str] = Field(
        default_factory=list,
        description="Spec rules or inferred invariants this defect violates",
    )


class TriageResult(BaseModel):
    defects: list[Defect]
