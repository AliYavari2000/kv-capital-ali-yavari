"""Tools for Node 8 - Normalization (LLM agent).

Standardizes subject and comp data so adjustments apply consistently. All
conversions and scale mappings are deterministic; the agent orchestrates them,
documents assumptions, and escalates material missing data.
"""

from __future__ import annotations

from typing import Any, Optional

from src import valuation_math as vm
from src.tools.base import SHARED_TOOL_SPECS, ToolkitBase, fn_spec

_CONDITION_SCALE = {"poor": 1, "fair": 2, "average": 3, "good": 4, "excellent": 5, "renovated": 6}
_QUALITY_SCALE = {"economy": 1, "standard": 2, "custom": 3, "luxury": 4}
_AREA_TO_SQFT = {"sqft": 1.0, "sqm": 10.7639, "acres": 43560.0, "hectares": 107639.0}


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


def _basement(v: Any) -> str:
    t = str(v or "").lower()
    if "walk" in t:
        return "walkout"
    if "unfin" in t:
        return "unfinished"
    if "fin" in t or "develop" in t:
        return "finished"
    if t in ("none", "no", "0"):
        return "none"
    return "standardized"


def _garage_spaces(v: Any) -> int:
    t = str(v or "").lower()
    if "triple" in t or "3" in t:
        return 3
    if "double" in t or "2" in t:
        return 2
    if "single" in t or "1" in t:
        return 1
    if "no garage" in t or t in ("none", "no", "0"):
        return 0
    return 0


class NormalizationToolkit(ToolkitBase):
    node_name = "normalization"

    def __init__(self, state: dict[str, Any]) -> None:
        super().__init__()
        self.subject = dict(state.get("subject", {}))
        self.comps = [dict(c) for c in state.get("ranked_comps", [])]
        self.assumptions: list[str] = []
        self.outliers: list[dict[str, Any]] = []
        self._normalized = False

    def _norm_record(self, rec: dict[str, Any]) -> dict[str, Any]:
        out = dict(rec)
        if out.get("gla_sqft") is not None:
            out["gla_sqft"] = _to_int(out["gla_sqft"])
        if out.get("lot_size_sqft") is not None:
            out["lot_size_sqft"] = _to_int(out["lot_size_sqft"])
        if out.get("bathrooms") is not None:
            bf = _to_float(out["bathrooms"])
            out["bathrooms"] = round(bf * 2) / 2.0 if bf is not None else None
        if out.get("sale_date"):
            out["sale_date"] = str(out["sale_date"])[:10]
        cond = str(out.get("condition") or self.subject.get("condition") or "average").lower()
        out["condition"] = cond.title()
        out["condition_score"] = _CONDITION_SCALE.get(cond, 3)
        qual = str(out.get("quality") or self.subject.get("quality") or "standard").lower()
        out["quality_score"] = _QUALITY_SCALE.get(qual, 2)
        out["basement"] = _basement(out.get("basement") or self.subject.get("basement"))
        out["garage_spaces"] = _garage_spaces(out.get("garage") or self.subject.get("garage"))
        return out

    def canonical_property_mapper(self) -> dict[str, Any]:
        """Map subject + comps to the standard schema and normalize all features."""
        self.subject = self._norm_record(self.subject)
        self.comps = [self._norm_record(c) for c in self.comps]
        self._normalized = True
        return {"subject_fields": sorted(self.subject.keys()), "comps_normalized": len(self.comps)}

    def unit_convert(self, value: float, from_unit: str = "sqft", to_unit: str = "sqft") -> dict[str, Any]:
        """Convert an area value between units (deterministic)."""
        v = _to_float(value)
        if v is None:
            return {"value": None}
        sqft = v * _AREA_TO_SQFT.get(from_unit.lower(), 1.0)
        result = sqft / _AREA_TO_SQFT.get(to_unit.lower(), 1.0)
        return {"value": round(result, 2), "from": from_unit, "to": to_unit}

    def date_normalizer(self, date: str) -> dict[str, Any]:
        """Normalize a date to ISO (YYYY-MM-DD)."""
        try:
            return {"date": vm._parse_date(date).isoformat()}
        except Exception:
            return {"date": str(date)[:10]}

    def gla_normalizer(self, value: float, unit: str = "sqft") -> dict[str, Any]:
        """Standardize a GLA value to above-grade square feet."""
        return self.unit_convert(value, unit, "sqft")

    def basement_normalizer(self, value: str) -> dict[str, Any]:
        """Encode basement consistently (finished/unfinished/walkout/none)."""
        return {"basement": _basement(value)}

    def garage_normalizer(self, value: str) -> dict[str, Any]:
        """Standardize garage to a space count."""
        return {"garage_spaces": _garage_spaces(value)}

    def condition_scale_mapper(self, condition: str) -> dict[str, Any]:
        """Map a condition label to the 1-6 numeric scale."""
        return {"condition_score": _CONDITION_SCALE.get(str(condition).lower(), 3)}

    def quality_scale_mapper(self, quality: str) -> dict[str, Any]:
        """Map a build-quality label to the 1-4 numeric scale."""
        return {"quality_score": _QUALITY_SCALE.get(str(quality).lower(), 2)}

    def location_feature_encoder(self, features: Optional[dict] = None) -> dict[str, Any]:
        """Encode location features (backing, busy road, view, corner lot)."""
        enc = features or self.subject.get("site_features") or {}
        self.subject["location_features"] = enc
        return {"location_features": enc}

    def missing_value_handler(self) -> dict[str, Any]:
        """Default unstated comp features to the subject's so they net to zero."""
        filled = 0
        for c in self.comps:
            for fld in ("condition", "basement", "garage_spaces", "quality_score"):
                if c.get(fld) in (None, ""):
                    c[fld] = self.subject.get(fld)
                    filled += 1
        return {"filled": filled}

    def outlier_feature_detector(self) -> dict[str, Any]:
        """Flag unusual feature values among comps."""
        outliers = []
        for c in self.comps:
            gla = _to_float(c.get("gla_sqft")) or 0
            if gla and (gla < 400 or gla > 8000):
                outliers.append({"id": c.get("id"), "field": "gla_sqft", "value": gla})
        self.outliers = outliers
        return {"outliers": outliers}

    def normalization_report_writer(self, assumptions: Optional[list] = None) -> dict[str, Any]:
        """Document normalization assumptions and transformations."""
        self.assumptions = assumptions or []
        return {"assumptions": self.assumptions}


TOOL_SPECS = [
    fn_spec("canonical_property_mapper", "Map subject + comps to the standard schema and normalize features."),
    fn_spec("unit_convert", "Convert an area value between units.",
            {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
            ["value"]),
    fn_spec("date_normalizer", "Normalize a date to ISO.", {"date": {"type": "string"}}, ["date"]),
    fn_spec("gla_normalizer", "Standardize a GLA value to above-grade square feet.",
            {"value": {"type": "number"}, "unit": {"type": "string"}}, ["value"]),
    fn_spec("basement_normalizer", "Encode basement (finished/unfinished/walkout/none).",
            {"value": {"type": "string"}}, ["value"]),
    fn_spec("garage_normalizer", "Standardize garage to a space count.",
            {"value": {"type": "string"}}, ["value"]),
    fn_spec("condition_scale_mapper", "Map a condition label to the 1-6 numeric scale.",
            {"condition": {"type": "string"}}, ["condition"]),
    fn_spec("quality_scale_mapper", "Map a build-quality label to the 1-4 numeric scale.",
            {"quality": {"type": "string"}}, ["quality"]),
    fn_spec("location_feature_encoder", "Encode location features (backing, busy road, view, corner lot).",
            {"features": {"type": "object"}}),
    fn_spec("missing_value_handler", "Default unstated comp features to the subject's."),
    fn_spec("outlier_feature_detector", "Flag unusual feature values among comps."),
    fn_spec("normalization_report_writer", "Document normalization assumptions.",
            {"assumptions": {"type": "array", "items": {"type": "string"}}}),
] + SHARED_TOOL_SPECS


def build_dispatch(tk: NormalizationToolkit) -> dict[str, Any]:
    d = {
        "canonical_property_mapper": tk.canonical_property_mapper,
        "unit_convert": tk.unit_convert,
        "date_normalizer": tk.date_normalizer,
        "gla_normalizer": tk.gla_normalizer,
        "basement_normalizer": tk.basement_normalizer,
        "garage_normalizer": tk.garage_normalizer,
        "condition_scale_mapper": tk.condition_scale_mapper,
        "quality_scale_mapper": tk.quality_scale_mapper,
        "location_feature_encoder": tk.location_feature_encoder,
        "missing_value_handler": tk.missing_value_handler,
        "outlier_feature_detector": tk.outlier_feature_detector,
        "normalization_report_writer": tk.normalization_report_writer,
    }
    d.update(tk.shared_dispatch())
    return d
