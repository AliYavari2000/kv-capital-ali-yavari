"""Tools for Node 5 - Market Scope (LLM agent).

The agent defines the comp-search envelope (area, time window, property-type and
feature filters, market segment). Geometry helpers (radius, drive-time->radius)
are deterministic; the agent chooses the parameters and rationale.
"""

from __future__ import annotations

from typing import Any, Optional

from src import case_store, config
from src.tools.base import SHARED_TOOL_SPECS, ToolkitBase, fn_spec

_ADJACENT = {
    "Detached": ["Detached", "Semi-Detached"],
    "Semi-Detached": ["Semi-Detached", "Detached", "Townhouse"],
    "Townhouse": ["Townhouse", "Semi-Detached", "Condo"],
    "Condo": ["Condo", "Townhouse"],
}
# Rough Calgary/Edmonton residential drive-time -> radius (km).
_DRIVE_TIME_KM = {5: 2.0, 10: 4.0, 15: 6.0}


class MarketScopeToolkit(ToolkitBase):
    node_name = "market_scope"

    def __init__(self, state: dict[str, Any]) -> None:
        super().__init__()
        self.case_dir = state.get("case_dir")
        s = state.get("subject", {})
        self.ptype = s.get("property_type")
        self.gla = float(s.get("gla_sqft") or 0)
        self.neighborhood = s.get("neighborhood")
        self.city = s.get("city")
        self.scope: dict[str, Any] = {}
        self.filters: dict[str, Any] = {}

    def neighbourhood_boundary_lookup(self) -> dict[str, Any]:
        """Identify the subject neighbourhood/community boundary."""
        community = None
        if self.case_dir:
            cb = case_store.load_case(self.case_dir).zoning()["data"].get("community_boundary")
            if isinstance(cb, dict) and cb.get("features"):
                community = cb["features"][0].get("properties", {}).get("community")
        return {"neighbourhood": community or self.neighborhood, "city": self.city}

    def submarket_classifier(self) -> dict[str, Any]:
        """Assign the subject to a price-tier market segment."""
        ppsf = config.NEIGHBORHOODS.get(self.neighborhood, {}).get("base_ppsf", 0)
        if ppsf >= 600:
            seg = "premium"
        elif ppsf >= 480:
            seg = "mid-market"
        else:
            seg = "entry"
        self.scope["segment"] = seg
        return {"segment": seg, "base_ppsf": ppsf}

    def radius_search_area(self, radius_km: float) -> dict[str, Any]:
        """Set the geographic search radius (km)."""
        self.scope["radius_km"] = round(float(radius_km), 2)
        return {"radius_km": self.scope["radius_km"]}

    def drive_time_polygon(self, minutes: int = 10) -> dict[str, Any]:
        """Approximate a drive-time search area as an equivalent radius."""
        km = _DRIVE_TIME_KM.get(int(minutes), 4.0)
        self.scope["radius_km"] = km
        self.scope["drive_time_min"] = int(minutes)
        return {"minutes": minutes, "approx_radius_km": km}

    def amenity_distance_calculator(self) -> dict[str, Any]:
        """Representative distances to parks/schools/transit/commercial (km)."""
        return {"parks_km": 0.4, "schools_km": 0.8, "transit_km": 0.6, "commercial_km": 1.2}

    def market_inventory_snapshot(self) -> dict[str, Any]:
        """Pull active-listing / absorption counts for the area."""
        if not self.case_dir:
            return {"active_listings": None, "months_of_supply": None}
        m = case_store.load_case(self.case_dir).market()["data"]
        absorb = m.get("absorption_inventory") or []
        snap = absorb[-1] if isinstance(absorb, list) and absorb else {}
        out = {"active_listings": snap.get("active_listings"),
               "sales": snap.get("sales"),
               "months_of_supply": snap.get("months_of_supply")}
        self.scope["inventory"] = out
        return out

    def search_window_selector(self, months: int) -> dict[str, Any]:
        """Select the sales time window in months (e.g. 3, 6, 12)."""
        self.scope["recency_months"] = int(months)
        return {"recency_months": int(months)}

    def comp_filter_builder(self, gla_range: Optional[list] = None,
                            bedrooms_range: Optional[list] = None,
                            bathrooms_range: Optional[list] = None,
                            property_types: Optional[list] = None,
                            same_neighbourhood_preferred: bool = True) -> dict[str, Any]:
        """Build comp filters (property types, GLA/bed/bath ranges)."""
        ptypes = property_types or _ADJACENT.get(self.ptype, [self.ptype] if self.ptype else [])
        filters = {
            "property_type": self.ptype,
            "property_types": ptypes,
            "gla_range": gla_range,
            "bedrooms_range": bedrooms_range,
            "bathrooms_range": bathrooms_range,
            "same_neighbourhood_preferred": same_neighbourhood_preferred,
        }
        self.filters = filters
        self.scope["property_types"] = ptypes
        # Translate GLA range -> band fraction the retrieval layer uses.
        if gla_range and self.gla > 0 and len(gla_range) == 2:
            lo, hi = float(gla_range[0]), float(gla_range[1])
            band = max(abs(hi - self.gla), abs(self.gla - lo)) / self.gla
            self.scope["gla_band"] = round(band, 3)
        return filters

    def search_expansion_policy(self, expand: bool = False, reason: str = "") -> dict[str, Any]:
        """Record a policy to widen area/time if comps are thin."""
        self.scope["expansion"] = {"expand": expand, "reason": reason}
        return {"expand": expand, "reason": reason}


TOOL_SPECS = [
    fn_spec("neighbourhood_boundary_lookup", "Identify the subject neighbourhood/community boundary."),
    fn_spec("submarket_classifier", "Assign the subject to a price-tier market segment."),
    fn_spec("radius_search_area", "Set the geographic search radius (km).",
            {"radius_km": {"type": "number"}}, ["radius_km"]),
    fn_spec("drive_time_polygon", "Approximate a drive-time area (5/10/15 min) as a radius.",
            {"minutes": {"type": "integer", "enum": [5, 10, 15]}}),
    fn_spec("amenity_distance_calculator", "Distances to parks/schools/transit/commercial."),
    fn_spec("market_inventory_snapshot", "Pull active-listing / absorption counts for the area."),
    fn_spec("search_window_selector", "Select the sales time window in months (e.g. 3, 6, 12).",
            {"months": {"type": "integer"}}, ["months"]),
    fn_spec("comp_filter_builder", "Build comp filters (property types, GLA/bed/bath ranges).",
            {"gla_range": {"type": "array", "items": {"type": "number"}},
             "bedrooms_range": {"type": "array", "items": {"type": "number"}},
             "bathrooms_range": {"type": "array", "items": {"type": "number"}},
             "property_types": {"type": "array", "items": {"type": "string"}},
             "same_neighbourhood_preferred": {"type": "boolean"}}),
    fn_spec("search_expansion_policy", "Record a policy to widen area/time if comps are thin.",
            {"expand": {"type": "boolean"}, "reason": {"type": "string"}}),
] + SHARED_TOOL_SPECS


def build_dispatch(tk: MarketScopeToolkit) -> dict[str, Any]:
    d = {
        "neighbourhood_boundary_lookup": tk.neighbourhood_boundary_lookup,
        "submarket_classifier": tk.submarket_classifier,
        "radius_search_area": tk.radius_search_area,
        "drive_time_polygon": tk.drive_time_polygon,
        "amenity_distance_calculator": tk.amenity_distance_calculator,
        "market_inventory_snapshot": tk.market_inventory_snapshot,
        "search_window_selector": tk.search_window_selector,
        "comp_filter_builder": tk.comp_filter_builder,
        "search_expansion_policy": tk.search_expansion_policy,
    }
    d.update(tk.shared_dispatch())
    return d
