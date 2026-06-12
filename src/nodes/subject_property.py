"""Node 2 - Subject Property (``SubjectPropertyNode``).

Inspects and measures the subject correctly. Canonicalizes property type and
neighborhood, coerces beds/baths/GLA/lot/year, derives effective age, captures
condition and upgrade notes, attaches lat/lon (neighborhood or city centroid),
and runs the subject data-quality check that gates the rest of the pipeline.

Tools: listing parser, image/measurement rules, plausibility validators.
"""

from __future__ import annotations

from typing import Any, Optional

from src import config
from src.state import CompState

_TYPE_SYNONYMS = {
    "detached": "Detached", "single family": "Detached", "single-family": "Detached",
    "house": "Detached", "bungalow": "Detached",
    "semi": "Semi-Detached", "semi detached": "Semi-Detached", "semi-detached": "Semi-Detached",
    "duplex": "Semi-Detached",
    "townhouse": "Townhouse", "town house": "Townhouse", "town home": "Townhouse",
    "townhome": "Townhouse", "row house": "Townhouse", "rowhouse": "Townhouse",
    "condo": "Condo", "condominium": "Condo", "apartment": "Condo", "apt": "Condo",
}

# Plausibility bounds for sanity checks.
_BOUNDS = {
    "gla_sqft": (300, 12000),
    "bedrooms": (0, 10),
    "bathrooms": (0.5, 12),
    "lot_size_sqft": (0, 60000),
    "year_built": (1890, config.VALUATION_DATE.year),
}


def _canon_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = str(value).strip().lower()
    if v in _TYPE_SYNONYMS:
        return _TYPE_SYNONYMS[v]
    for key, canon in _TYPE_SYNONYMS.items():
        if key in v:
            return canon
    for t in config.PROPERTY_TYPES:
        if t.lower() == v:
            return t
    return None


def _resolve_neighborhood(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = str(value).strip().lower()
    for nb in config.NEIGHBORHOODS:
        if nb.lower() == v:
            return nb
    for nb in config.NEIGHBORHOODS:
        if v in nb.lower() or nb.lower() in v:
            return nb
    return value  # keep as-is; coords fall back to city centroid


def _city_centroid(city: str) -> Optional[tuple[float, float]]:
    pts = [(m["lat"], m["lon"]) for m in config.NEIGHBORHOODS.values() if m["city"] == city]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(str(v).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def _data_quality(s: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    missing_critical = [f for f in config.CRITICAL_FIELDS if s.get(f) in (None, "")]
    missing_soft = [f for f in config.SOFT_FIELDS if s.get(f) in (None, "")]

    if s.get("lat") is None or s.get("lon") is None:
        missing_critical.append("coordinates")

    for fld, (lo, hi) in _BOUNDS.items():
        v = s.get(fld)
        if v is not None and not (lo <= float(v) <= hi):
            issues.append(f"{fld}={v} outside plausible range [{lo}, {hi}]")

    crit_ok = len(config.CRITICAL_FIELDS) - len([f for f in missing_critical if f in config.CRITICAL_FIELDS])
    crit_frac = crit_ok / len(config.CRITICAL_FIELDS)
    soft_ok = len(config.SOFT_FIELDS) - len(missing_soft)
    soft_frac = soft_ok / len(config.SOFT_FIELDS)
    score = round(0.7 * crit_frac + 0.3 * soft_frac, 3)
    if issues:
        score = round(max(0.0, score - 0.1 * len(issues)), 3)

    passed = len(missing_critical) == 0 and not issues
    return {
        "score": score,
        "passed": passed,
        "missing_critical": missing_critical,
        "missing_soft": missing_soft,
        "issues": issues,
    }


def subject_property_node(state: CompState) -> dict[str, Any]:
    s = dict(state.get("subject", {}))
    notes: list[str] = []

    s["property_type"] = _canon_type(s.get("property_type"))

    nb = _resolve_neighborhood(s.get("neighborhood"))
    s["neighborhood"] = nb
    if nb in config.NEIGHBORHOODS and not s.get("city"):
        s["city"] = config.NEIGHBORHOODS[nb]["city"]

    for fld in ("bedrooms", "gla_sqft", "lot_size_sqft", "year_built"):
        if s.get(fld) is not None:
            val = _to_float(s[fld])
            s[fld] = int(val) if val is not None else None
    if s.get("bathrooms") is not None:
        s["bathrooms"] = _to_float(s["bathrooms"])

    # Condition / upgrades: keep what was provided, default to market-typical so
    # the adjustment grid treats the subject neutrally when unstated.
    s["condition"] = (s.get("condition") or "Average").title() if s.get("condition") else "Average"
    if s.get("upgrades") is None:
        s["upgrades"] = []

    # Coordinates
    if s.get("lat") is None or s.get("lon") is None:
        if nb in config.NEIGHBORHOODS:
            s["lat"] = config.NEIGHBORHOODS[nb]["lat"]
            s["lon"] = config.NEIGHBORHOODS[nb]["lon"]
            notes.append("coords from neighborhood centroid")
        elif s.get("city"):
            c = _city_centroid(s["city"])
            if c:
                s["lat"], s["lon"] = c
                notes.append("coords from city centroid (approx)")

    # Effective age
    if s.get("year_built"):
        s["property_age"] = max(0, config.VALUATION_DATE.year - int(s["year_built"]))

    dq = _data_quality(s)

    trace = state.get("trace", []) + [
        f"subject_property: type={s.get('property_type')}, neighborhood={s.get('neighborhood')}, "
        f"city={s.get('city')}, condition={s.get('condition')}"
        + (f" [{'; '.join(notes)}]" if notes else "")
        + f"; data_quality score={dq['score']}, passed={dq['passed']}, "
        f"missing_critical={dq['missing_critical'] or 'none'}"
    ]
    return {"subject": s, "data_quality": dq, "trace": trace}
