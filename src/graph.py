"""LangGraph wiring for the comp-analysis agent.

Standard residential-appraisal pipeline with a data-quality gate and a
human-in-the-loop final sign-off:

    assignment_intake -> subject_property
        subject_property --(critical field missing)--> human_review
        subject_property --(ok)--> legal_title -> zoning_hbu -> market_scope
            -> comp_retrieval -> fact_verification -> normalization
            -> adjustment_engine -> reconciliation
                reconciliation --(low confidence)--> human_review
                reconciliation --(ok)--> report_writer
        human_review --(reject + valued + under limit)--> market_scope  (re-pull)
        human_review --(else)--> report_writer
    report_writer -> END

A checkpointer is attached so the ``interrupt`` in human_review can pause and
later resume on the same thread.
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.nodes.node1_assignment_intake import assignment_intake_node
from src.nodes.node2_subject_property import subject_property_node
from src.nodes.node3_legal_title import legal_title_node
from src.nodes.node4_zoning_hbu import zoning_hbu_node
from src.nodes.node5_market_scope import market_scope_node
from src.nodes.node6_comp_retrieval import comp_retrieval_node
from src.nodes.node7_fact_verification import fact_verification_node
from src.nodes.node8_normalization import normalization_node
from src.nodes.node9_adjustment_engine import adjustment_engine_node
from src.nodes.node10_reconciliation import reconciliation_node
from src.nodes.node11_report_writer import report_writer_node
from src.nodes.human_review import human_review_node
from src.state import CompState, new_state

_MAX_RERUNS = 1


def _route_after_subject(state: CompState) -> str:
    dq = state.get("data_quality", {})
    return "legal_title" if dq.get("passed", False) else "human_review"


def _route_after_reconciliation(state: CompState) -> str:
    return "human_review" if state.get("risk", {}).get("requires_human_review") else "report_writer"


def _route_after_human(state: CompState) -> str:
    decision = state.get("human_decision", {})
    valued = bool(state.get("valuation")) and state["valuation"].get("method") != "no_comps"
    if decision.get("action") == "reject" and valued and state.get("rerun_count", 0) <= _MAX_RERUNS:
        return "market_scope"
    return "report_writer"


def build_graph(checkpointer: Optional[Any] = None):
    """Build and compile the agent graph. Pass a checkpointer (defaults to an
    in-memory saver) to enable human-in-the-loop interrupt/resume."""
    g = StateGraph(CompState)

    g.add_node("assignment_intake", assignment_intake_node)
    g.add_node("subject_property", subject_property_node)
    g.add_node("legal_title", legal_title_node)
    g.add_node("zoning_hbu", zoning_hbu_node)
    g.add_node("market_scope", market_scope_node)
    g.add_node("comp_retrieval", comp_retrieval_node)
    g.add_node("fact_verification", fact_verification_node)
    g.add_node("normalization", normalization_node)
    g.add_node("adjustment_engine", adjustment_engine_node)
    g.add_node("reconciliation", reconciliation_node)
    g.add_node("human_review", human_review_node)
    g.add_node("report_writer", report_writer_node)

    g.add_edge(START, "assignment_intake")
    g.add_edge("assignment_intake", "subject_property")
    g.add_conditional_edges("subject_property", _route_after_subject,
                            {"legal_title": "legal_title", "human_review": "human_review"})
    g.add_edge("legal_title", "zoning_hbu")
    g.add_edge("zoning_hbu", "market_scope")
    g.add_edge("market_scope", "comp_retrieval")
    g.add_edge("comp_retrieval", "fact_verification")
    g.add_edge("fact_verification", "normalization")
    g.add_edge("normalization", "adjustment_engine")
    g.add_edge("adjustment_engine", "reconciliation")
    g.add_conditional_edges("reconciliation", _route_after_reconciliation,
                            {"human_review": "human_review", "report_writer": "report_writer"})
    g.add_conditional_edges("human_review", _route_after_human,
                            {"market_scope": "market_scope", "report_writer": "report_writer"})
    g.add_edge("report_writer", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


def run_headless(raw_input: Any, *, human_decision: Optional[dict] = None,
                 thread_id: str = "cli") -> CompState:
    """Run the full graph to completion without pausing.

    If the deal triggers human review, ``human_decision`` is used as the
    reviewer's verdict (defaults to approve), so the run never blocks. Returns
    the final state.
    """
    app = build_graph()
    state = new_state(raw_input)
    # Inject a decision so the interrupt in human_review never blocks headless runs.
    state["human_decision"] = human_decision or {
        "action": "approve", "reviewer": "headless", "note": "Auto-approved (headless run)."
    }
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(state, config=config)
