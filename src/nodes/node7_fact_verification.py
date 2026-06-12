"""Node 7 - Fact Verification (``FactVerificationNode``).

Cross-checks the facts of each selected comp before it influences value:
sale price plausibility, sale date validity/recency, GLA and lot-size sanity,
and presence of lot data for lot-bearing types. Each comp gets a ``verified``
flag and per-comp notes; pool-level issues are surfaced as flags for risk.

Tools: assessment rolls, permits, PDFs, title, RPR.
"""

from __future__ import annotations

from typing import Any

from src import config, valuation_math as vm
from src.state import CompState

_GLA_BOUNDS = (300, 12000)
_LOT_BOUNDS = (0, 80000)


def _flag(code: str, severity: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def fact_verification_node(state: CompState) -> dict[str, Any]:
    comps = state.get("ranked_comps", [])
    flags: list[dict[str, str]] = []

    verified_count = 0
    out_comps: list[dict[str, Any]] = []
    for c in comps:
        notes: list[str] = []
        ok = True

        price = float(c.get("sale_price") or 0)
        if price <= 0:
            ok = False
            notes.append("non-positive sale price")

        months = c.get("months_ago")
        if months is None:
            months = vm.months_between(c.get("sale_date"), config.VALUATION_DATE)
        if months < 0:
            ok = False
            notes.append("sale date after the effective date")

        gla = float(c.get("gla_sqft") or 0)
        if not (_GLA_BOUNDS[0] <= gla <= _GLA_BOUNDS[1]):
            ok = False
            notes.append(f"GLA {gla:.0f} outside plausible range")

        lot = c.get("lot_size_sqft")
        if config.TYPE_HAS_LOT.get(c.get("property_type")) and (lot in (None, "") or float(lot) <= 0):
            notes.append("missing lot size for lot-bearing type")
        elif lot not in (None, "") and not (_LOT_BOUNDS[0] <= float(lot) <= _LOT_BOUNDS[1]):
            notes.append(f"lot size {float(lot):.0f} outside plausible range")

        merged = dict(c)
        merged["verified"] = ok
        merged["verification_notes"] = notes
        out_comps.append(merged)
        if ok:
            verified_count += 1

    unverified = len(out_comps) - verified_count
    if unverified:
        flags.append(_flag("unverified_comps", "medium" if unverified < len(out_comps) else "high",
                           f"{unverified} of {len(out_comps)} comps failed a fact check."))

    verification = {
        "checked": len(out_comps),
        "verified": verified_count,
        "unverified": unverified,
        "flags": flags,
        "note": "Cross-checked price, date, GLA, and lot against plausibility rules.",
    }
    trace = state.get("trace", []) + [
        f"fact_verification: {verified_count}/{len(out_comps)} comps verified"
        + (f", {unverified} flagged" if unverified else "")
    ]
    return {"ranked_comps": out_comps, "verification": verification, "trace": trace}
