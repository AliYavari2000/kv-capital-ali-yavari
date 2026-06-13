"""Node 7 - Fact Verification (LLM agent).

Verifies subject and comp facts across MLS / assessment / permit / title / RPR
sources, scoring confidence and flagging conflicts. Plausibility checks and
confidence math are deterministic; the agent decides what to verify, records
verified facts, and escalates unresolved conflicts.

No deterministic fallback: requires an LLM.
"""

from __future__ import annotations

import json
from typing import Any

from src import llm
from src import fast_path
from src.state import CompState
from src.tools import verification_tools

_SYSTEM_PROMPT = (
    "You are the Fact Verification agent for KV Capital. Using ONLY the tools, "
    "verify the subject and comparable facts across sources. Do not assert facts "
    "you did not verify.\n\n"
    "Steps: cross_source_fact_checker on the comps; assessment_record_lookup and "
    "permit_lookup to corroborate the subject; sale_price_verifier; the "
    "gla/lot/property_type conflict detectors; for the key subject facts (e.g. "
    "gla_sqft, year_built, lot_size_sqft) call confidence_score then "
    "verified_fact_writer with the value, confidence, and sources. append_evidence "
    "for each source. Call raise_human_review for unresolved conflicts. Finish "
    "with a one-line verification summary."
)


def fact_verification_node(state: CompState) -> dict[str, Any]:
    if not llm.llm_available():
        raise llm.LLMUnavailableError(
            "Fact Verification is an LLM agent and requires OPENAI_API_KEY to be set."
        )

    tk = verification_tools.VerificationToolkit(state)
    dispatch = verification_tools.build_dispatch(tk)
    user = ("Verify facts for the subject and these comps:\n"
            + json.dumps({"subject": {k: tk.subject.get(k) for k in
                                      ("gla_sqft", "lot_size_sqft", "year_built", "property_type")},
                          "comp_ids": [c.get("id") for c in tk.comps]}, indent=2, default=str))
    run, mode = llm.run_node_agent(
        state, dispatch, verification_tools.TOOL_SPECS, _SYSTEM_PROMPT, user,
        lambda: fast_path.run_verification(tk, dispatch),
    )

    # Guarantee comps carry a verified flag for downstream nodes.
    if not tk.summary:
        tk.cross_source_fact_checker()

    for esc in tk.escalations:
        tk.flags.append({"code": "verification_escalation", "severity": esc["severity"],
                         "message": esc["reason"]})

    verification = {
        "checked": tk.summary.get("checked", len(tk.comps)),
        "verified": tk.summary.get("verified", 0),
        "unverified": tk.summary.get("unverified", 0),
        "verified_facts": tk.verified_facts,
        "flags": tk.flags,
        "note": "Cross-checked comp facts against plausibility rules and corroborating sources.",
    }

    tools_used = [c["tool"] for c in run["calls"]]
    trace = state.get("trace", []) + [
        f"fact_verification ({mode}): {verification['verified']}/{verification['checked']} "
        f"comps verified, {len(tk.verified_facts)} subject facts written, "
        f"flags={len(tk.flags)}; tools={tools_used}"
    ]
    return {
        "ranked_comps": tk.comps,
        "verification": verification,
        "evidence": state.get("evidence", []) + tk.evidence,
        "trace": trace,
    }
