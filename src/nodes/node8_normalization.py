"""Node 8 - Normalization (LLM agent).

Standardizes units, dates, GLA, basement/garage, condition/quality scales, and
location features across the subject and comps so the adjustment grid compares
like with like. All conversions are deterministic; the agent orchestrates them
and documents the assumptions.

No deterministic fallback: requires an LLM.
"""

from __future__ import annotations

from typing import Any

from src import llm
from src.state import CompState
from src.tools import normalization_tools

_SYSTEM_PROMPT = (
    "You are the Normalization agent for KV Capital. Using ONLY the tools, "
    "standardize all subject and comp data so adjustments can be applied "
    "consistently.\n\n"
    "Steps: canonical_property_mapper to map and normalize everything; "
    "location_feature_encoder; missing_value_handler so unstated comp features "
    "default to the subject; outlier_feature_detector; then "
    "normalization_report_writer to record your assumptions. append_evidence for "
    "assumptions, and raise_human_review for material missing data. Finish with a "
    "one-line summary."
)


def normalization_node(state: CompState) -> dict[str, Any]:
    if not llm.llm_available():
        raise llm.LLMUnavailableError(
            "Normalization is an LLM agent and requires OPENAI_API_KEY to be set."
        )

    tk = normalization_tools.NormalizationToolkit(state)
    dispatch = normalization_tools.build_dispatch(tk)
    user = (f"Normalize the subject and {len(tk.comps)} comps. "
            "Call canonical_property_mapper first, then handle missing values and "
            "outliers, and document assumptions.")
    run = llm.run_tool_agent(_SYSTEM_PROMPT, user, normalization_tools.TOOL_SPECS, dispatch)

    # Guarantee normalization happened for downstream math.
    if not tk._normalized:
        tk.canonical_property_mapper()
        tk.missing_value_handler()

    tools_used = [c["tool"] for c in run["calls"]]
    trace = state.get("trace", []) + [
        f"normalization (LLM agent): standardized subject + {len(tk.comps)} comps "
        f"(GLA/lot/baths/dates, condition/quality scores, basement/garage), "
        f"outliers={len(tk.outliers)}, assumptions={len(tk.assumptions)}; tools={tools_used}"
    ]
    return {
        "subject": tk.subject,
        "ranked_comps": tk.comps,
        "normalization": {"assumptions": tk.assumptions, "outliers": tk.outliers},
        "evidence": state.get("evidence", []) + tk.evidence,
        "trace": trace,
    }
