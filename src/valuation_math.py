"""Deterministic, auditable math for the comp-analysis agent.

This module holds the explainable core: geographic distance, similarity
scoring, and the sales-comparison adjustment grid. None of these functions call
an LLM -- every number a loan officer sees can be traced back to a formula and
the coefficients in ``src/config.py``.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

from src import config


# ---------------------------------------------------------------------------
# Geo + time helpers
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def months_between(earlier: Any, later: Any) -> float:
    d1, d2 = _parse_date(earlier), _parse_date(later)
    return (d2.year - d1.year) * 12 + (d2.month - d1.month) + (d2.day - d1.day) / 30.0


def base_ppsf(neighborhood: str | None, city: str | None = None) -> float:
    """Neighborhood baseline detached $/sqft, with a city-median fallback."""
    if neighborhood and neighborhood in config.NEIGHBORHOODS:
        return float(config.NEIGHBORHOODS[neighborhood]["base_ppsf"])
    if city:
        vals = [m["base_ppsf"] for m in config.NEIGHBORHOODS.values() if m["city"] == city]
        if vals:
            return float(sorted(vals)[len(vals) // 2])
    allv = [m["base_ppsf"] for m in config.NEIGHBORHOODS.values()]
    return float(sorted(allv)[len(allv) // 2])


def effective_ppsf(neighborhood: str | None, property_type: str, city: str | None = None) -> float:
    mult = config.TYPE_PPSF_MULTIPLIER.get(property_type, 1.0)
    return base_ppsf(neighborhood, city) * mult


# ---------------------------------------------------------------------------
# Similarity scoring
# ---------------------------------------------------------------------------
def _half_life_score(value: float, half: float) -> float:
    """1.0 at value 0, ~0.5 at value == half, decaying toward 0."""
    if half <= 0:
        return 1.0 if value == 0 else 0.0
    return float(0.5 ** (abs(value) / half))


def similarity(subject: dict[str, Any], comp: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Overall similarity in [0,1] plus a per-factor breakdown."""
    w = config.RANKING_WEIGHTS

    # Distance
    dist_km = comp.get("distance_km")
    if dist_km is None:
        dist_km = haversine_km(subject["lat"], subject["lon"], comp["lat"], comp["lon"])
    s_dist = _half_life_score(dist_km, config.DISTANCE_HALF_KM)

    # Recency
    months_ago = comp.get("months_ago")
    if months_ago is None:
        months_ago = months_between(comp["sale_date"], config.VALUATION_DATE)
    s_recency = _half_life_score(months_ago, config.RECENCY_HALF_MONTHS)

    # GLA (relative)
    subj_gla = max(1.0, float(subject.get("gla_sqft") or 0))
    gla_frac = abs(float(comp["gla_sqft"]) - subj_gla) / subj_gla
    s_gla = _half_life_score(gla_frac, config.GLA_HALF_FRACTION)

    # Beds + baths
    dbeds = abs(float(comp.get("bedrooms", 0)) - float(subject.get("bedrooms", comp.get("bedrooms", 0))))
    dbaths = abs(float(comp.get("bathrooms", 0)) - float(subject.get("bathrooms", comp.get("bathrooms", 0))))
    s_bb = _half_life_score(dbeds + dbaths, 1.5)

    # Property type (partial credit for adjacent tiers via multiplier closeness)
    ms = config.TYPE_PPSF_MULTIPLIER.get(subject.get("property_type"), 1.0)
    mc = config.TYPE_PPSF_MULTIPLIER.get(comp.get("property_type"), 1.0)
    s_type = 1.0 if subject.get("property_type") == comp.get("property_type") else max(0.0, 1.0 - abs(ms - mc) * 4)

    # Neighborhood / city
    if comp.get("neighborhood") == subject.get("neighborhood"):
        s_nb = 1.0
    elif comp.get("city") == subject.get("city"):
        s_nb = 0.4
    else:
        s_nb = 0.0

    breakdown = {
        "distance": s_dist,
        "recency": s_recency,
        "gla": s_gla,
        "beds_baths": s_bb,
        "same_type": s_type,
        "same_neighborhood": s_nb,
    }
    total_w = sum(w.values())
    overall = sum(breakdown[k] * w[k] for k in w) / total_w
    return overall, breakdown


# ---------------------------------------------------------------------------
# Sales-comparison adjustments
# ---------------------------------------------------------------------------
def _line(factor: str, detail: str, amount: float) -> dict[str, Any]:
    return {"factor": factor, "detail": detail, "amount": round(float(amount), 0)}


def compute_adjustments(subject: dict[str, Any], comp: dict[str, Any]) -> dict[str, Any]:
    """Adjust a comp's sale price toward the subject (in valuation-date dollars).

    Returns a dict with itemized ``adjustments`` lines, ``gross_adjustment``,
    ``net_adjustment``, and ``adjusted_price``. All adjustments bring the comp
    *to* the subject: a positive amount means the comp is inferior and is
    adjusted upward.
    """
    lines: list[dict[str, Any]] = []
    sale_price = float(comp["sale_price"])

    subj_nb = subject.get("neighborhood")
    subj_city = subject.get("city")
    subj_type = subject.get("property_type")
    subj_mult = config.TYPE_PPSF_MULTIPLIER.get(subj_type, 1.0)
    subj_bppsf = base_ppsf(subj_nb, subj_city)
    subj_eppsf = subj_bppsf * subj_mult

    # 1) Time / market: bring the comp's price to the valuation date.
    months_ago = comp.get("months_ago")
    if months_ago is None:
        months_ago = months_between(comp["sale_date"], config.VALUATION_DATE)
    time_factor = (1.0 + config.MONTHLY_APPRECIATION) ** months_ago
    time_adj = sale_price * (time_factor - 1.0)
    lines.append(_line(
        "Time/Market",
        f"Sold {months_ago:.1f} mo ago; +{config.MONTHLY_APPRECIATION*100:.1f}%/mo appreciation",
        time_adj,
    ))

    # 2) Property type: convert comp's structure tier to the subject's tier.
    comp_type = comp.get("property_type")
    comp_mult = config.TYPE_PPSF_MULTIPLIER.get(comp_type, 1.0)
    if comp_type != subj_type:
        type_adj = (subj_mult - comp_mult) * float(comp["gla_sqft"]) * subj_bppsf
        lines.append(_line(
            "Property Type",
            f"{comp_type} -> {subj_type}",
            type_adj,
        ))

    # 3) GLA (gross living area)
    gla_delta = float(subject.get("gla_sqft", comp["gla_sqft"])) - float(comp["gla_sqft"])
    if abs(gla_delta) >= 1:
        gla_adj = gla_delta * subj_eppsf
        lines.append(_line(
            "GLA",
            f"{gla_delta:+.0f} sqft @ ${subj_eppsf:,.0f}/sqft",
            gla_adj,
        ))

    # 4) Bedrooms
    if subject.get("bedrooms") is not None:
        dbeds = float(subject["bedrooms"]) - float(comp.get("bedrooms", subject["bedrooms"]))
        if abs(dbeds) >= 1:
            lines.append(_line("Bedrooms", f"{dbeds:+.0f} bed @ ${config.BED_VALUE:,}", dbeds * config.BED_VALUE))

    # 5) Bathrooms
    if subject.get("bathrooms") is not None:
        dbaths = float(subject["bathrooms"]) - float(comp.get("bathrooms", subject["bathrooms"]))
        if abs(dbaths) >= 0.5:
            lines.append(_line("Bathrooms", f"{dbaths:+.1f} bath @ ${config.BATH_VALUE:,}", dbaths * config.BATH_VALUE))

    # 6) Lot size (only when both properties carry private lot value)
    if config.TYPE_HAS_LOT.get(subj_type) and config.TYPE_HAS_LOT.get(comp_type):
        if subject.get("lot_size_sqft") is not None and comp.get("lot_size_sqft") is not None:
            dlot = float(subject["lot_size_sqft"]) - float(comp["lot_size_sqft"])
            if abs(dlot) >= 50:
                lines.append(_line("Lot Size", f"{dlot:+.0f} sqft @ ${config.LOT_PPSF}/sqft", dlot * config.LOT_PPSF))

    # 7) Effective age (year built)
    if subject.get("year_built") and comp.get("year_built"):
        subj_age = min(config.VALUATION_DATE.year - int(subject["year_built"]), config.MAX_AGE_FOR_DEPRECIATION)
        comp_age = min(config.VALUATION_DATE.year - int(comp["year_built"]), config.MAX_AGE_FOR_DEPRECIATION)
        dage = comp_age - subj_age  # subject newer (smaller age) -> positive
        if abs(dage) >= 1:
            lines.append(_line("Age", f"{dage:+.0f} yr effective age @ ${config.AGE_VALUE_PER_YEAR:,}/yr", dage * config.AGE_VALUE_PER_YEAR))

    net = sum(line["amount"] for line in lines)
    gross = sum(abs(line["amount"]) for line in lines)
    adjusted_price = sale_price + net
    return {
        "adjustments": lines,
        "net_adjustment": round(net, 0),
        "gross_adjustment": round(gross, 0),
        "adjusted_price": round(adjusted_price, 0),
        "gross_adjustment_pct": round(gross / sale_price, 4) if sale_price else 0.0,
    }
