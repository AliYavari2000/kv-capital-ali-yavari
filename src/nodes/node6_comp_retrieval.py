"""Node 6 - Comp Retrieval (``CompRetrievalNode``).

Pulls recent sold comparables from the store using the market scope (radius,
recency, GLA band, property-type filter) with progressive widening, scores each
candidate against the subject by weighted similarity, selects the top-N working
set, and attaches active + pending/conditional listing context for the market.

Tools: MLS / IDX / private DB / public sources.
"""

from __future__ import annotations

from typing import Any

from src import config, data_store, valuation_math as vm
from src.state import CompState

_CARRY = [
    "id", "address", "neighborhood", "city", "property_type", "bedrooms",
    "bathrooms", "gla_sqft", "lot_size_sqft", "year_built", "sale_date",
    "sale_price", "distance_km", "months_ago",
]


def comp_retrieval_node(state: CompState) -> dict[str, Any]:
    subject = state.get("subject", {})
    scope = state.get("market_scope", {})

    result = data_store.retrieve_candidates(subject, scope=scope)
    candidates = result["candidates"]
    meta = result["meta"]

    # Score + select the top-N most comparable sold comps.
    scored: list[dict[str, Any]] = []
    for c in candidates:
        overall, breakdown = vm.similarity(subject, c)
        row = {k: c.get(k) for k in _CARRY}
        row["similarity"] = round(overall, 4)
        row["similarity_breakdown"] = {k: round(v, 3) for k, v in breakdown.items()}
        scored.append(row)
    scored.sort(key=lambda r: r["similarity"], reverse=True)
    top = scored[: config.TOP_N_COMPS]

    market = data_store.market_context(candidates)

    final = meta["final_filters"]
    trace = state.get("trace", []) + [
        f"comp_retrieval: {meta['candidate_count']} sold candidates "
        f"(radius={final['radius_km']}km, recency={final['recency_months']}mo, "
        f"gla_band=+/-{int(final['gla_band']*100)}%, widened={meta['widened']}); "
        f"selected top {len(top)}; market context {len(market.get('active', []))} active / "
        f"{len(market.get('pending', []))} pending"
        + (
            ", top sims = " + ", ".join(f"{r['id']}:{r['similarity']:.2f}" for r in top)
            if top else ""
        )
    ]
    return {
        "candidates": candidates,
        "retrieval_meta": meta,
        "ranked_comps": top,
        "market_context": market,
        "trace": trace,
    }
