"""Top-level LangGraph wiring.

ground → plan → [HITL ambiguity gate] → generate → static_check
      → [HITL review gate] ─ approve → execute (Executor) → triage (Auditor) → report
                           ├ revise  → generate            (feedback loop)
                           ├ edited  → static_check        (re-validate human edits)
                           └ abort   → report

Grounding reads only the live OpenAPI spec; the review gate lets a QE verify,
exclude, edit, or send back generated tests before anything executes.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from qe_agent.report import node_report
from qe_agent.stages.execution import node_execute, node_review_gate, route_after_review
from qe_agent.stages.generation import node_generate, node_static_check
from qe_agent.stages.planning import node_ambiguity_gate, node_ground, node_plan
from qe_agent.stages.triage import node_triage
from qe_agent.state import QEState


def build_graph(checkpointer=None):
    graph = StateGraph(QEState)
    graph.add_node("ground", node_ground)
    graph.add_node("plan", node_plan)
    graph.add_node("ambiguity_gate", node_ambiguity_gate)
    graph.add_node("generate", node_generate)
    graph.add_node("static_check", node_static_check)
    graph.add_node("review_gate", node_review_gate)
    graph.add_node("execute", node_execute)
    graph.add_node("triage", node_triage)
    graph.add_node("report", node_report)

    graph.add_edge(START, "ground")
    graph.add_edge("ground", "plan")
    graph.add_edge("plan", "ambiguity_gate")
    graph.add_edge("ambiguity_gate", "generate")
    graph.add_edge("generate", "static_check")
    graph.add_edge("static_check", "review_gate")
    graph.add_conditional_edges(
        "review_gate", route_after_review, ["execute", "generate", "static_check", "report"]
    )
    graph.add_edge("execute", "triage")
    graph.add_edge("triage", "report")
    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
