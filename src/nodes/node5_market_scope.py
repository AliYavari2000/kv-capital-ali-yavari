"""Node 5 - Market Scope (``MarketScopeNode``).

Sets the comp-search area and property-type filter before any data is pulled:
search radius, recency (time) window, GLA band, and which property types are
acceptable comparables for this subject. The chosen scope parametrizes the
retrieval step (and is re-entered on a reviewer re-pull).

Tools: GIS, neighbourhood boundaries, drive-time.
"""

from __future__ import annotations

from typing import Any

from src import config
from src.state import CompState

# Property types that bracket a subject type when same-type comps are thin.
_ADJACENT = {
    "Detached": ["Detached", "Semi-Detached"],
    "Semi-Detached": ["Semi-Detached", "Detached", "Townhouse"],
    "Townhouse": ["Townhouse", "Semi-Detached", "Condo"],
    "Condo": ["Condo", "Townhouse"],
}


def market_scope_node(state: CompState) -> dict[str, Any]:
    s = state.get("subject", {})
    p = config.RETRIEVAL
    ptype = s.get("property_type")
    rerun = state.get("rerun_count", 0)

    radius = float(p["radius_km"])
    recency = int(p["recency_months"])
    gla_band = float(p["gla_band"])
    notes: list[str] = []

    # Dense, attached-housing types support a tighter geographic scope; detached
    # subjects often need a wider net.
    if ptype in ("Condo", "Townhouse"):
        radius = round(radius * 0.7, 2)
        notes.append("tightened radius for attached housing")
    elif ptype == "Detached":
        notes.append("standard detached radius")

    # On a reviewer re-pull, broaden the envelope up front.
    if rerun:
        radius = round(radius * config.RETRIEVAL["widen_radius_factor"], 2)
        recency += config.RETRIEVAL["widen_recency_add_months"]
        gla_band = round(gla_band + config.RETRIEVAL["widen_gla_band_add"], 2)
        notes.append(f"broadened on re-pull #{rerun}")

    property_types = _ADJACENT.get(ptype, [ptype] if ptype else list(config.PROPERTY_TYPES))

    rationale = (
        f"Search within {radius} km of the subject, sales in the last {recency} months, "
        f"GLA +/-{int(gla_band*100)}%, comparable types {property_types}."
        + (f" ({'; '.join(notes)})" if notes else "")
    )

    market_scope = {
        "radius_km": radius,
        "recency_months": recency,
        "gla_band": gla_band,
        "property_types": property_types,
        "rationale": rationale,
    }
    trace = state.get("trace", []) + [
        f"market_scope: radius={radius}km, recency={recency}mo, gla_band=+/-{int(gla_band*100)}%, "
        f"types={property_types}"
    ]
    return {"market_scope": market_scope, "trace": trace}
