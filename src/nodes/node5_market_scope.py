"""Node 5 - Market Scope (LLM agent).

Defines the comp-search envelope: area (radius / drive-time), time window,
property-type and feature filters, and market segment. Geometry is deterministic;
the agent chooses parameters and rationale, and the resulting scope parametrizes
retrieval.

No deterministic fallback: requires an LLM.
"""

from __future__ import annotations

import json
from typing import Any

from src import config, llm
from src.state import CompState
from src.tools import market_scope_tools as mst

_SYSTEM_PROMPT = (
    "You are the Market Scope agent for KV Capital. Using ONLY the tools, define "
    "where and how to search for comparable sales for the subject.\n\n"
    "Steps: neighbourhood_boundary_lookup; submarket_classifier; "
    "market_inventory_snapshot; choose a search area with radius_search_area or "
    "drive_time_polygon; choose a time window with search_window_selector "
    "(tighter for active markets, wider for thin ones); comp_filter_builder with "
    "sensible GLA/bed/bath ranges around the subject; set search_expansion_policy "
    "if the market looks thin. append_evidence for why you chose the area, and "
    "raise_human_review for unusual or very thin markets. Finish with a one-line "
    "rationale."
)


def market_scope_node(state: CompState) -> dict[str, Any]:
    if not llm.llm_available():
        raise llm.LLMUnavailableError(
            "Market Scope is an LLM agent and requires OPENAI_API_KEY to be set."
        )

    tk = mst.MarketScopeToolkit(state)
    dispatch = mst.build_dispatch(tk)
    rerun = state.get("rerun_count", 0)
    user = ("Define the market scope for this subject:\n"
            + json.dumps({"property_type": tk.ptype, "gla_sqft": tk.gla,
                          "neighborhood": tk.neighborhood, "city": tk.city,
                          "rerun_count": rerun}, indent=2)
            + ("\nThis is a re-pull; broaden the envelope." if rerun else ""))
    run = llm.run_tool_agent(_SYSTEM_PROMPT, user, mst.TOOL_SPECS, dispatch)

    p = config.RETRIEVAL
    scope = tk.scope
    radius = float(scope.get("radius_km", p["radius_km"]))
    recency = int(scope.get("recency_months", p["recency_months"]))
    gla_band = float(scope.get("gla_band", p["gla_band"]))
    property_types = scope.get("property_types") or mst._ADJACENT.get(
        tk.ptype, [tk.ptype] if tk.ptype else list(config.PROPERTY_TYPES))

    # On a reviewer re-pull, broaden up front regardless of the agent's choice.
    if rerun:
        radius = round(radius * p["widen_radius_factor"], 2)
        recency += p["widen_recency_add_months"]
        gla_band = round(gla_band + p["widen_gla_band_add"], 2)

    market_scope = {
        "radius_km": radius,
        "recency_months": recency,
        "gla_band": gla_band,
        "property_types": property_types,
        "segment": scope.get("segment"),
        "filters": tk.filters,
        "inventory": scope.get("inventory"),
        "rationale": (f"Search within {radius} km, last {recency} months, GLA +/-"
                      f"{int(gla_band*100)}%, types {property_types}."),
    }

    tools_used = [c["tool"] for c in run["calls"]]
    trace = state.get("trace", []) + [
        f"market_scope (LLM agent): radius={radius}km, recency={recency}mo, "
        f"gla_band=+/-{int(gla_band*100)}%, segment={scope.get('segment')}, "
        f"types={property_types}; tools={tools_used}"
    ]
    return {
        "market_scope": market_scope,
        "documents": dict(state.get("documents", {})),
        "evidence": state.get("evidence", []) + tk.evidence,
        "trace": trace,
    }
