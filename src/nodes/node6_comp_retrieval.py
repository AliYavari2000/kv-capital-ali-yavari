"""Node 6 - Comp Retrieval (LLM agent).

Retrieves a broad candidate pool of sold comps (plus active/pending context),
scores similarity, de-duplicates, flags non-arm's-length and outlier sales,
ranks, and selects the strongest working set, logging rejections with reasons.
Retrieval/similarity math is deterministic; the agent orchestrates and judges.

No deterministic fallback: requires an LLM.
"""

from __future__ import annotations

import json
from typing import Any

from src import config, data_store, llm
from src.state import CompState
from src.tools import retrieval_tools

_SYSTEM_PROMPT = (
    "You are the Comp Retrieval agent for KV Capital. Using ONLY the tools, build "
    "a strong set of comparable sales for the subject. Retrieve MORE candidates "
    "than needed, then narrow down.\n\n"
    "Steps: query_sold_comps; similarity_search_comps; deduplicate_properties; "
    "arms_length_filter and outlier_sale_detector (log any rejects with "
    "comp_rejection_logger and a reason); comp_candidate_ranker to select the "
    "strongest working set; query_active_listings and query_pending_or_conditional "
    "for market context. append_evidence for the comp source. Call "
    "raise_human_review if too few usable comps are found. Finish with a one-line "
    "summary of how many comps you selected."
)


def comp_retrieval_node(state: CompState) -> dict[str, Any]:
    if not llm.llm_available():
        raise llm.LLMUnavailableError(
            "Comp Retrieval is an LLM agent and requires OPENAI_API_KEY to be set."
        )

    tk = retrieval_tools.RetrievalToolkit(state)
    dispatch = retrieval_tools.build_dispatch(tk)
    user = ("Retrieve comparables for this subject using the market scope:\n"
            + json.dumps({"subject": {k: tk.subject.get(k) for k in
                                      ("property_type", "gla_sqft", "neighborhood", "city")},
                          "scope": tk.scope}, indent=2, default=str))
    run = llm.run_tool_agent(_SYSTEM_PROMPT, user, retrieval_tools.TOOL_SPECS, dispatch)

    # Deterministic guarantees: ensure a pool, scoring, ranking, and context exist.
    if not tk.candidates:
        tk.query_sold_comps()
    if not tk.scored:
        tk.similarity_search_comps()
    if not tk.ranked:
        tk.comp_candidate_ranker(config.TOP_N_COMPS)
    if not tk.market:
        tk.market = data_store.market_context(tk.candidates, case_dir=tk.case_dir)

    tools_used = [c["tool"] for c in run["calls"]]
    final = tk.meta.get("final_filters", {})
    trace = state.get("trace", []) + [
        f"comp_retrieval (LLM agent): {tk.meta.get('candidate_count', len(tk.candidates))} "
        f"candidates (radius={final.get('radius_km')}km, recency={final.get('recency_months')}mo, "
        f"widened={tk.meta.get('widened')}); selected {len(tk.ranked)}, rejected {len(tk.rejected)}; "
        f"tools={tools_used}"
    ]
    return {
        "candidates": tk.candidates,
        "retrieval_meta": tk.meta,
        "ranked_comps": tk.ranked,
        "rejected_comps": tk.rejected,
        "market_context": tk.market,
        "evidence": state.get("evidence", []) + tk.evidence,
        "trace": trace,
    }
