"""Eval harness: how we test the tester.

Scores pipeline output (defects) against the labeled ground truth in
sut/bugs.yaml, and probes the agents' resistance to prompt injection with
poisoned OpenAPI spec variants. Lives entirely OUTSIDE the pipeline — the
agents never see this package, and this package never changes agent behavior.
"""
