"""Node 8 - Feature Normalization (``NormalizationNode``).

Standardizes measurements and features across the subject and the selected comps
so the adjustment grid compares like with like: numeric coercion of GLA / lot,
half-step rounding of bathrooms, consistent basement / garage / condition
treatment, and defaulting unstated features to the subject's so they net to zero.

Tools: Python validators.
"""

from __future__ import annotations

from typing import Any, Optional

from src.state import CompState


def _to_int(v: Any) -> Optional[int]:
    try:
        return int(round(float(str(v).replace(",", "").replace("$", "").strip())))
    except (TypeError, ValueError):
        return None


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(str(v).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def _half_step(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return round(v * 2) / 2.0


def _normalize_features(rec: dict[str, Any], subject: dict[str, Any]) -> dict[str, Any]:
    out = dict(rec)
    if out.get("gla_sqft") is not None:
        out["gla_sqft"] = _to_int(out["gla_sqft"])
    if out.get("lot_size_sqft") is not None:
        out["lot_size_sqft"] = _to_int(out["lot_size_sqft"])
    if out.get("bathrooms") is not None:
        out["bathrooms"] = _half_step(_to_float(out["bathrooms"]))

    # Condition / basement / garage are standardized; when unknown they default
    # to the subject so the adjustment engine treats them as neutral.
    out["condition"] = (out.get("condition") or subject.get("condition") or "Average")
    out["basement"] = out.get("basement") or subject.get("basement") or "Standardized"
    out["garage"] = out.get("garage") or subject.get("garage") or "Standardized"
    return out


def normalization_node(state: CompState) -> dict[str, Any]:
    subject = dict(state.get("subject", {}))
    comps = state.get("ranked_comps", [])

    # Standardize the subject's own feature set first.
    subject["condition"] = subject.get("condition") or "Average"
    subject["basement"] = subject.get("basement") or "Standardized"
    subject["garage"] = subject.get("garage") or "Standardized"

    normalized = [_normalize_features(c, subject) for c in comps]

    trace = state.get("trace", []) + [
        f"normalization: standardized units/features on subject + {len(normalized)} comps "
        f"(GLA, lot, baths half-step, condition/basement/garage)"
    ]
    return {"subject": subject, "ranked_comps": normalized, "trace": trace}
