"""Node 4 - Zoning, Land-Use, Legality, Highest-and-Best-Use (LLM agent).

Determines legal permissibility and concludes highest-and-best-use via the four
classic tests (legally permissible, physically possible, financially feasible,
maximally productive). The checks are rule-based tools; the agent applies them
and writes the explanation.

No deterministic fallback: requires an LLM.
"""

from __future__ import annotations

import json
from typing import Any

from src import llm
from src.state import CompState
from src.tools import zoning_tools

_SYSTEM_PROMPT = (
    "You are the Zoning / Land-Use / Highest-and-Best-Use agent for KV Capital. "
    "Using ONLY the tools, determine legal permissibility and conclude HBU. "
    "Do not invent zoning codes.\n\n"
    "Steps: municipal_zoning_lookup and zoning_map_query; permitted_use_checker; "
    "development_standard_extractor; nonconforming_use_detector; overlay_lookup; "
    "permit_history_lookup; then apply hbu_rule_engine, supplying your boolean "
    "conclusion for each of the four tests (legally permissible, physically "
    "possible, financially feasible, maximally productive) based on the evidence; "
    "finally zoning_summary_generator with a concise explanation. append_evidence "
    "for sources, and raise_human_review on zoning conflicts (e.g. nonconforming "
    "use, restrictive overlay)."
)


def _flag(code: str, severity: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def zoning_hbu_node(state: CompState) -> dict[str, Any]:
    if not llm.llm_available():
        raise llm.LLMUnavailableError(
            "Zoning/HBU is an LLM agent and requires OPENAI_API_KEY to be set."
        )

    tk = zoning_tools.ZoningToolkit(state)
    dispatch = zoning_tools.build_dispatch(tk)
    user = ("Assess zoning and HBU for this subject:\n"
            + json.dumps({"property_type": tk.ptype, "lot_size_sqft": tk.lot}, indent=2))
    run = llm.run_tool_agent(_SYSTEM_PROMPT, user, zoning_tools.TOOL_SPECS, dispatch)

    rec = tk.record
    flags: list[dict[str, str]] = []
    if rec.get("zoning_code", "Unverified") == "Unverified":
        flags.append(_flag("zoning_unverified", "medium",
                           "Property type unknown; zoning and permitted use could not be determined."))
    if rec.get("nonconforming"):
        flags.append(_flag("nonconforming_lot", "medium",
                           "Current use may be legal nonconforming under current zoning."))
    for o in rec.get("overlays", []):
        flags.append(_flag("overlay", "medium", f"Subject affected by overlay: {o}."))
    for esc in tk.escalations:
        flags.append(_flag("zoning_escalation", esc["severity"], esc["reason"]))

    conforming = not rec.get("nonconforming", False) and rec.get("zoning_code", "Unverified") != "Unverified"
    hbu = rec.get("summary") or (
        "Present use (as-improved residential) is the highest and best use."
        if rec.get("hbu_as_improved", conforming) else
        "Present use as improved, subject to confirming legal nonconforming status.")

    zoning = {
        "zoning_code": rec.get("zoning_code", "Unverified"),
        "permitted_use": rec.get("permitted_use", "Unverified"),
        "conforming": conforming,
        "highest_and_best_use": hbu,
        "hbu_tests": tk.hbu_tests,
        "development_standards": rec.get("development_standards"),
        "overlays": rec.get("overlays", []),
        "permits": rec.get("permits"),
        "flags": flags,
    }

    tools_used = [c["tool"] for c in run["calls"]]
    trace = state.get("trace", []) + [
        f"zoning_hbu (LLM agent): zoning={zoning['zoning_code']}, conforming={conforming}, "
        f"hbu_tests={tk.hbu_tests or 'n/a'}, flags={len(flags)}; tools={tools_used}"
    ]
    return {
        "zoning": zoning,
        "documents": dict(state.get("documents", {})),
        "evidence": state.get("evidence", []) + tk.evidence,
        "trace": trace,
    }
