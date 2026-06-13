"""Node 12 - Human Reviewer / Final Sign-off (human-in-the-loop).

When reconciliation flags Low confidence (or subject data quality failed), the
graph pauses here using LangGraph's ``interrupt`` and waits for a reviewer's
sign-off:

    approve  -> accept the reconciled estimate
    override -> substitute a reviewer's value
    reject   -> send the deal back to re-scope + re-pull comps (once)

When sign-off is not required, the node auto-approves and passes straight to the
report. A decision supplied up front in state (e.g. for headless/CLI runs) is
honored without pausing.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from langgraph.types import interrupt

from src.state import CompState

_MAX_RERUNS = 1


def _gather_escalations(state: CompState) -> list[dict[str, Any]]:
    """Collect agent escalations raised upstream (intake/legal/zoning + risk)."""
    esc: list[dict[str, Any]] = []
    esc.extend(state.get("assignment", {}).get("escalations", []) or [])
    return esc


def _build_payload(state: CompState) -> dict[str, Any]:
    risk = state.get("risk", {})
    valuation = state.get("valuation", {})
    dq = state.get("data_quality", {})
    return {
        "reason": "Confidence is Low" if risk.get("confidence") == "Low" else "Data quality check failed",
        "confidence": risk.get("confidence"),
        "model_estimate": valuation.get("point_estimate"),
        "range": [valuation.get("low"), valuation.get("high")],
        "flags": risk.get("flags", []),
        "escalations": _gather_escalations(state),
        "data_quality": dq,
        "subject": state.get("subject", {}),
        "comps": [{"id": c.get("id"), "adjusted_price": c.get("adjusted_price"),
                   "weight": c.get("weight")} for c in
                  (state.get("adjusted_comps") or state.get("ranked_comps", []))],
        "options": ["approve", "override", "reject"],
    }


def _audit_entry(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "node": "human_review",
        "source": "reviewer_decision",
        "detail": (f"{decision.get('action', 'approve')} by "
                   f"{decision.get('reviewer', 'analyst')}: {decision.get('note', '')}"),
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
    }


def _apply(state: CompState, decision: dict[str, Any]) -> dict[str, Any]:
    action = decision.get("action", "approve")
    valuation = dict(state.get("valuation", {}))
    out: dict[str, Any] = {"human_decision": decision}

    if action == "override" and decision.get("override_value") is not None:
        valuation["model_estimate"] = valuation.get("point_estimate")
        ov = float(decision["override_value"])
        valuation["point_estimate"] = round(ov, -2)
        valuation["low"] = round(ov * 0.97, -2)
        valuation["high"] = round(ov * 1.03, -2)
        gla = float(state.get("subject", {}).get("gla_sqft") or 0)
        valuation["implied_ppsf"] = round(ov / gla, 0) if gla else valuation.get("implied_ppsf", 0)
        valuation["method"] = "human_override"
        out["valuation"] = valuation
    elif action == "reject":
        out["rerun_count"] = state.get("rerun_count", 0) + 1

    return out


def human_review_node(state: CompState) -> dict[str, Any]:
    risk = state.get("risk", {})
    dq = state.get("data_quality", {})
    requires = risk.get("requires_human_review", False) or not dq.get("passed", True)

    if not requires:
        decision = {"action": "approve", "note": "Auto-approved: sufficient confidence, no human review required.",
                    "reviewer": "system"}
        trace = state.get("trace", []) + ["human_review: not required -> auto-approved"]
        return {"human_decision": decision,
                "evidence": state.get("evidence", []) + [_audit_entry(decision)],
                "trace": trace}

    # If a decision was injected up front (headless runs), honor it without pausing.
    decision = state.get("human_decision")
    if not decision:
        decision = interrupt(_build_payload(state))

    # Guard against reject loops.
    if decision.get("action") == "reject" and state.get("rerun_count", 0) >= _MAX_RERUNS:
        decision = {**decision, "action": "approve",
                    "note": "Re-pull limit reached; proceeding with model estimate. " + decision.get("note", "")}

    out = _apply(state, decision)
    action = out["human_decision"].get("action")
    out["evidence"] = state.get("evidence", []) + [_audit_entry(decision)]
    trace = state.get("trace", []) + [
        f"human_review: reviewer={decision.get('reviewer', 'analyst')} action={action}"
        + (f" override=${decision.get('override_value'):,.0f}" if action == "override" and decision.get("override_value") else "")
    ]
    out["trace"] = trace
    return out
