"""Node 4 - Zoning + Highest-and-Best-Use (``ZoningHBUNode``).

Reviews zoning, land-use controls, legality of the current use, and concludes
highest-and-best-use (HBU). Maps the subject's property type to a representative
Alberta residential zoning district, checks whether the current use conforms,
and flags nonconforming or transitional situations.

Tools: municipal zoning APIs, bylaws, GIS.
"""

from __future__ import annotations

from typing import Any

from src.state import CompState

# Representative residential zoning by property type (Calgary/Edmonton style).
_ZONING_BY_TYPE = {
    "Detached": {"code": "R-C1 / RF1", "permitted": "Single detached dwelling"},
    "Semi-Detached": {"code": "R-C2 / RF3", "permitted": "Semi-detached / duplex dwelling"},
    "Townhouse": {"code": "R-CG / RF5", "permitted": "Row / townhouse dwelling"},
    "Condo": {"code": "M-C1 / RA7", "permitted": "Multi-residential (apartment)"},
}


def _flag(code: str, severity: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def zoning_hbu_node(state: CompState) -> dict[str, Any]:
    s = state.get("subject", {})
    ptype = s.get("property_type")
    flags: list[dict[str, str]] = []

    z = _ZONING_BY_TYPE.get(ptype)
    if z is None:
        zoning_code = "Unverified"
        permitted_use = "Unverified"
        conforming = False
        flags.append(_flag("zoning_unverified", "medium",
                           "Property type unknown; zoning and permitted use could not be determined."))
    else:
        zoning_code = z["code"]
        permitted_use = z["permitted"]
        conforming = True

    # Lot-size heuristic: a detached home on a very small lot may be a legal
    # nonconforming / redevelopment situation worth noting.
    lot = s.get("lot_size_sqft")
    if ptype == "Detached" and lot is not None and float(lot) < 3000:
        conforming = False
        flags.append(_flag("nonconforming_lot", "medium",
                           f"Detached use on a {int(lot)} sqft lot may be legal nonconforming under current zoning."))

    # HBU conclusion: for standard residential subjects, present use is HBU.
    if conforming:
        hbu = "Present use (as-improved residential) is the highest and best use."
    else:
        hbu = ("Present use as improved, subject to confirming legal nonconforming status; "
               "redevelopment potential to be considered.")

    zoning = {
        "zoning_code": zoning_code,
        "permitted_use": permitted_use,
        "conforming": conforming,
        "highest_and_best_use": hbu,
        "flags": flags,
    }

    trace = state.get("trace", []) + [
        f"zoning_hbu: zoning={zoning_code}, conforming={conforming}, flags={len(flags)}"
    ]
    return {"zoning": zoning, "trace": trace}
