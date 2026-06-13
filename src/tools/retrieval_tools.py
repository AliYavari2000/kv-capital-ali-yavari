"""Tools for Node 6 - Comp Retrieval (LLM agent).

The agent retrieves a broad candidate pool, scores similarity, removes
duplicates, flags non-arm's-length and outlier sales, ranks, and selects the
strongest working set -- logging rejections with reasons. Retrieval and
similarity math are deterministic (``data_store`` / ``valuation_math``); the
agent orchestrates and judges.
"""

from __future__ import annotations

import statistics
from typing import Any

from src import config, data_store, valuation_math as vm
from src.tools.base import SHARED_TOOL_SPECS, ToolkitBase, fn_spec

_CARRY = [
    "id", "address", "neighborhood", "city", "property_type", "bedrooms",
    "bathrooms", "gla_sqft", "lot_size_sqft", "year_built", "sale_date",
    "sale_price", "distance_km", "months_ago",
]


class RetrievalToolkit(ToolkitBase):
    node_name = "comp_retrieval"

    def __init__(self, state: dict[str, Any]) -> None:
        super().__init__()
        self.case_dir = state.get("case_dir")
        self.subject = state.get("subject", {})
        self.scope = state.get("market_scope", {})
        self.candidates: list[dict[str, Any]] = []
        self.scored: list[dict[str, Any]] = []
        self.ranked: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []
        self.market: dict[str, Any] = {}
        self.meta: dict[str, Any] = {}

    def query_sold_comps(self) -> dict[str, Any]:
        """Pull recent sold comparables from the store using the market scope."""
        result = data_store.retrieve_candidates(self.subject, scope=self.scope,
                                                case_dir=self.case_dir)
        self.candidates = result["candidates"]
        self.meta = result["meta"]
        return {"candidate_count": len(self.candidates), "meta": self.meta}

    def geospatial_comp_search(self) -> dict[str, Any]:
        """Geographic/scope-based search (same pool as query_sold_comps)."""
        if not self.candidates:
            return self.query_sold_comps()
        return {"candidate_count": len(self.candidates)}

    def query_active_listings(self) -> dict[str, Any]:
        """Pull active listings for market context."""
        self.market = data_store.market_context(self.candidates, case_dir=self.case_dir)
        return {"active": self.market.get("active", [])}

    def query_pending_or_conditional(self) -> dict[str, Any]:
        """Pull conditional/pending sales for market context."""
        if not self.market:
            self.market = data_store.market_context(self.candidates, case_dir=self.case_dir)
        return {"pending": self.market.get("pending", [])}

    def similarity_search_comps(self) -> dict[str, Any]:
        """Score every candidate against the subject by weighted similarity."""
        scored = []
        for c in self.candidates:
            overall, breakdown = vm.similarity(self.subject, c)
            row = {k: c.get(k) for k in _CARRY}
            row["similarity"] = round(overall, 4)
            row["similarity_breakdown"] = {k: round(v, 3) for k, v in breakdown.items()}
            scored.append(row)
        scored.sort(key=lambda r: r["similarity"], reverse=True)
        self.scored = scored
        return {"scored": len(scored),
                "top": [{"id": r["id"], "similarity": r["similarity"]} for r in scored[:8]]}

    def deduplicate_properties(self) -> dict[str, Any]:
        """Remove duplicate MLS/public records by id/address."""
        seen, deduped, dropped = set(), [], 0
        for r in self.scored or self.candidates:
            key = (r.get("id") or r.get("address"))
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            deduped.append(r)
        if self.scored:
            self.scored = deduped
        return {"removed": dropped, "remaining": len(deduped)}

    def arms_length_filter(self) -> dict[str, Any]:
        """Flag possible non-arm's-length sales (price far below market $/sqft)."""
        flagged = []
        ppsfs = [r["sale_price"] / r["gla_sqft"] for r in self.scored
                 if r.get("sale_price") and r.get("gla_sqft")]
        if len(ppsfs) >= 3:
            med = statistics.median(ppsfs)
            for r in self.scored:
                if r.get("sale_price") and r.get("gla_sqft"):
                    ppsf = r["sale_price"] / r["gla_sqft"]
                    if ppsf < med * 0.6:
                        flagged.append(r["id"])
        return {"non_arms_length_suspects": flagged}

    def outlier_sale_detector(self) -> dict[str, Any]:
        """Flag suspiciously low/high sales by $/sqft dispersion."""
        flagged = []
        ppsfs = [r["sale_price"] / r["gla_sqft"] for r in self.scored
                 if r.get("sale_price") and r.get("gla_sqft")]
        if len(ppsfs) >= 4:
            mean = statistics.fmean(ppsfs)
            sd = statistics.pstdev(ppsfs)
            for r in self.scored:
                if r.get("sale_price") and r.get("gla_sqft") and sd:
                    z = (r["sale_price"] / r["gla_sqft"] - mean) / sd
                    if abs(z) > 2.5:
                        flagged.append({"id": r["id"], "z": round(z, 2)})
        return {"outliers": flagged}

    def comp_candidate_ranker(self, top_n: int = config.TOP_N_COMPS) -> dict[str, Any]:
        """Select the strongest comps for verification/use."""
        pool = self.scored or self.candidates
        self.ranked = pool[: int(top_n)]
        return {"selected": len(self.ranked),
                "ids": [r.get("id") for r in self.ranked]}

    def comp_rejection_logger(self, comp_id: str, reason: str) -> dict[str, Any]:
        """Record why a comp candidate was rejected."""
        self.rejected.append({"id": comp_id, "reason": reason})
        return {"rejected_count": len(self.rejected)}


TOOL_SPECS = [
    fn_spec("query_sold_comps", "Pull recent sold comparables using the market scope."),
    fn_spec("geospatial_comp_search", "Geographic/scope-based comp search."),
    fn_spec("query_active_listings", "Pull active listings for market context."),
    fn_spec("query_pending_or_conditional", "Pull conditional/pending sales for market context."),
    fn_spec("similarity_search_comps", "Score candidates against the subject by similarity."),
    fn_spec("deduplicate_properties", "Remove duplicate MLS/public records."),
    fn_spec("arms_length_filter", "Flag possible non-arm's-length sales."),
    fn_spec("outlier_sale_detector", "Flag suspiciously low/high sales."),
    fn_spec("comp_candidate_ranker", "Select the strongest comps (returns ranked working set).",
            {"top_n": {"type": "integer"}}),
    fn_spec("comp_rejection_logger", "Record why a comp candidate was rejected.",
            {"comp_id": {"type": "string"}, "reason": {"type": "string"}}, ["comp_id", "reason"]),
] + SHARED_TOOL_SPECS


def build_dispatch(tk: RetrievalToolkit) -> dict[str, Any]:
    d = {
        "query_sold_comps": tk.query_sold_comps,
        "geospatial_comp_search": tk.geospatial_comp_search,
        "query_active_listings": tk.query_active_listings,
        "query_pending_or_conditional": tk.query_pending_or_conditional,
        "similarity_search_comps": tk.similarity_search_comps,
        "deduplicate_properties": tk.deduplicate_properties,
        "arms_length_filter": tk.arms_length_filter,
        "outlier_sale_detector": tk.outlier_sale_detector,
        "comp_candidate_ranker": tk.comp_candidate_ranker,
        "comp_rejection_logger": tk.comp_rejection_logger,
    }
    d.update(tk.shared_dispatch())
    return d
