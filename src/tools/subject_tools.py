"""Tools for Node 2 - Subject Property Inspection / Measurement (LLM agent).

The agent extracts and classifies the subject's characteristics; the measurement
*math* (unit conversion, area normalization) and the schema/quality gate stay
deterministic. Photos and floor plans are exposed as document hooks until their
image/extraction tools land.
"""

from __future__ import annotations

from typing import Any, Optional

from src import case_store, config
from src.tools.base import SHARED_TOOL_SPECS, ToolkitBase, fn_spec

_TYPE_SYNONYMS = {
    "detached": "Detached", "single family": "Detached", "single-family": "Detached",
    "house": "Detached", "bungalow": "Detached",
    "semi": "Semi-Detached", "semi detached": "Semi-Detached", "semi-detached": "Semi-Detached",
    "duplex": "Semi-Detached",
    "townhouse": "Townhouse", "town house": "Townhouse", "town home": "Townhouse",
    "townhome": "Townhouse", "row house": "Townhouse", "rowhouse": "Townhouse",
    "condo": "Condo", "condominium": "Condo", "apartment": "Condo", "apt": "Condo",
}
_CONDITION_SCALE = ["poor", "fair", "average", "good", "excellent", "renovated"]
_QUALITY_SCALE = ["economy", "standard", "custom", "luxury"]
_BOUNDS = {
    "gla_sqft": (300, 12000), "bedrooms": (0, 10), "bathrooms": (0.5, 12),
    "lot_size_sqft": (0, 60000), "year_built": (1890, config.VALUATION_DATE.year),
}


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(str(v).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def normalize_lot_size(value: float, unit: str) -> Optional[float]:
    """Deterministic area conversion to square feet."""
    v = _to_float(value)
    if v is None:
        return None
    u = (unit or "sqft").strip().lower()
    if u in ("acre", "acres", "ac"):
        return v * 43560
    if u in ("sqm", "m2", "sq m", "square metres", "square meters"):
        return v * 10.7639
    if u in ("hectare", "hectares", "ha"):
        return v * 107639.0
    return v  # already sqft


class SubjectToolkit(ToolkitBase):
    node_name = "subject_property"

    def __init__(self, state: dict[str, Any]) -> None:
        super().__init__()
        self.case_dir = state.get("case_dir")
        self.subject: dict[str, Any] = dict(state.get("subject", {}))
        self.notes: list[str] = []
        self.conflicts: list[dict[str, Any]] = []
        self.data_quality: dict[str, Any] = {}
        self._documents = dict(state.get("documents", {}))

    # ------------------------------------------------------------------ tools
    def extract_subject_from_listing(self) -> dict[str, Any]:
        """Return the subject's raw listing facts already on file."""
        keys = ["address", "city", "neighborhood", "property_type", "bedrooms",
                "bathrooms", "gla_sqft", "lot_size_sqft", "year_built", "basement", "garage"]
        return {k: self.subject.get(k) for k in keys if self.subject.get(k) not in (None, "")}

    def inspect_photos(self) -> dict[str, Any]:
        """Subject photos (image-model extraction is a later tool)."""
        photos = self._documents.get("subject", {}).get("subject_photos", []) or []
        return {"photo_count": len(photos), "parsed": False,
                "note": "subject_photos present; image-model extraction pending."}

    def parse_floor_plan(self) -> dict[str, Any]:
        """Floor plan / RMS report (extraction is a later tool)."""
        docs = self._documents.get("subject", {})
        fp = docs.get("floor_plan", {})
        rms = docs.get("rms_measurement_report", {})
        return {"floor_plan_present": bool(fp.get("exists")),
                "rms_report_present": bool(rms.get("exists")),
                "parsed": False, "note": "floor plan / RMS extraction pending."}

    def normalize_measurement(self, value: float, unit: str = "sqft") -> dict[str, Any]:
        """Standardize an area measurement to square feet (deterministic)."""
        sqft = normalize_lot_size(value, unit)
        return {"value": value, "unit": unit, "sqft": round(sqft) if sqft is not None else None}

    def classify_property_type(self, property_type: str) -> dict[str, Any]:
        """Canonicalize the property type."""
        v = (property_type or "").strip().lower()
        canon = _TYPE_SYNONYMS.get(v)
        if not canon:
            for key, c in _TYPE_SYNONYMS.items():
                if key in v:
                    canon = c
                    break
        if not canon:
            for t in config.PROPERTY_TYPES:
                if t.lower() == v:
                    canon = t
        self.subject["property_type"] = canon
        return {"property_type": canon, "valid": bool(canon)}

    def classify_condition(self, condition: str) -> dict[str, Any]:
        """Record condition on the standard scale."""
        c = (condition or "").strip().lower()
        chosen = c if c in _CONDITION_SCALE else "average"
        self.subject["condition"] = chosen.title()
        return {"condition": chosen, "scale": _CONDITION_SCALE, "valid": c in _CONDITION_SCALE}

    def classify_quality(self, quality: str) -> dict[str, Any]:
        """Record build quality on the standard scale."""
        q = (quality or "").strip().lower()
        chosen = q if q in _QUALITY_SCALE else "standard"
        self.subject["quality"] = chosen
        return {"quality": chosen, "scale": _QUALITY_SCALE, "valid": q in _QUALITY_SCALE}

    def detect_renovations(self, renovations: Optional[list] = None) -> dict[str, Any]:
        """Record detected major upgrades."""
        ups = renovations or []
        if isinstance(ups, str):
            ups = [ups]
        self.subject["upgrades"] = ups
        return {"upgrades": ups, "count": len(ups)}

    def extract_site_features(self, features: Optional[dict] = None) -> dict[str, Any]:
        """Record site features (frontage, corner lot, backing, view, cul-de-sac)."""
        feats = features or {}
        self.subject["site_features"] = feats
        return {"site_features": feats}

    def check_measurement_conflicts(self) -> dict[str, Any]:
        """Compare listing GLA/lot against assessment and survey sources."""
        conflicts: list[dict[str, Any]] = []
        listing_gla = _to_float(self.subject.get("gla_sqft"))
        listing_lot = _to_float(self.subject.get("lot_size_sqft"))
        if self.case_dir:
            case = case_store.load_case(self.case_dir)
            # Assessment GLA/lot
            assess = (case.legal_title()["data"].get("assessment", {})
                      .get("assessment_open_data") or [])
            if isinstance(assess, list) and assess:
                a = assess[0]
                a_gla = _to_float(a.get("gla_sqft") or a.get("living_area"))
                if listing_gla and a_gla and abs(listing_gla - a_gla) / listing_gla > 0.10:
                    conflicts.append({"field": "gla_sqft", "listing": listing_gla,
                                      "assessment": a_gla})
            # Survey measurements
            surveys = case.zoning()["data"].get("permits", {}).get("survey_measurements") or []
            for row in surveys if isinstance(surveys, list) else []:
                label = str(row.get("label", "")).lower()
                val = _to_float(row.get("value_sqft"))
                if "lot" in label and listing_lot and val and abs(listing_lot - val) / listing_lot > 0.10:
                    conflicts.append({"field": "lot_size_sqft", "listing": listing_lot, "survey": val})
        self.conflicts = conflicts
        return {"conflicts": conflicts, "has_conflict": bool(conflicts)}

    def validate_subject_schema(self) -> dict[str, Any]:
        """Finalize coordinates/age + run the subject data-quality gate."""
        s = self.subject
        # Resolve neighborhood + coords.
        nb = s.get("neighborhood")
        if nb:
            for cand in config.NEIGHBORHOODS:
                if cand.lower() == str(nb).strip().lower():
                    nb = cand
                    break
            s["neighborhood"] = nb
            if nb in config.NEIGHBORHOODS and not s.get("city"):
                s["city"] = config.NEIGHBORHOODS[nb]["city"]
        for fld in ("bedrooms", "gla_sqft", "lot_size_sqft", "year_built"):
            if s.get(fld) is not None:
                v = _to_float(s[fld])
                s[fld] = int(v) if v is not None else None
        if s.get("bathrooms") is not None:
            s["bathrooms"] = _to_float(s["bathrooms"])
        s.setdefault("condition", "Average")
        s.setdefault("upgrades", [])
        if s.get("lat") is None or s.get("lon") is None:
            if nb in config.NEIGHBORHOODS:
                s["lat"], s["lon"] = config.NEIGHBORHOODS[nb]["lat"], config.NEIGHBORHOODS[nb]["lon"]
                self.notes.append("coords from neighborhood centroid")
            elif s.get("city"):
                pts = [(m["lat"], m["lon"]) for m in config.NEIGHBORHOODS.values()
                       if m["city"] == s["city"]]
                if pts:
                    s["lat"] = sum(p[0] for p in pts) / len(pts)
                    s["lon"] = sum(p[1] for p in pts) / len(pts)
                    self.notes.append("coords from city centroid (approx)")
        if s.get("year_built"):
            s["property_age"] = max(0, config.VALUATION_DATE.year - int(s["year_built"]))

        dq = self._data_quality(s)
        self.data_quality = dq
        return dq

    # --------------------------------------------------------------- helpers
    def _data_quality(self, s: dict[str, Any]) -> dict[str, Any]:
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
        soft_frac = (len(config.SOFT_FIELDS) - len(missing_soft)) / len(config.SOFT_FIELDS)
        score = round(0.7 * crit_frac + 0.3 * soft_frac, 3)
        if issues:
            score = round(max(0.0, score - 0.1 * len(issues)), 3)
        return {
            "score": score,
            "passed": len(missing_critical) == 0 and not issues,
            "missing_critical": missing_critical,
            "missing_soft": missing_soft,
            "issues": issues,
        }


TOOL_SPECS = [
    fn_spec("extract_subject_from_listing", "Return the subject's raw listing facts on file."),
    fn_spec("inspect_photos", "Check subject photos (image-model extraction pending)."),
    fn_spec("parse_floor_plan", "Check floor plan / RMS report (extraction pending)."),
    fn_spec("normalize_measurement", "Convert an area to square feet (sqft/sqm/acres/hectares).",
            {"value": {"type": "number"}, "unit": {"type": "string"}}, ["value"]),
    fn_spec("classify_property_type", "Canonicalize the property type.",
            {"property_type": {"type": "string"}}, ["property_type"]),
    fn_spec("classify_condition", "Classify condition: poor, fair, average, good, excellent, renovated.",
            {"condition": {"type": "string", "enum": _CONDITION_SCALE}}, ["condition"]),
    fn_spec("classify_quality", "Classify build quality: economy, standard, custom, luxury.",
            {"quality": {"type": "string", "enum": _QUALITY_SCALE}}, ["quality"]),
    fn_spec("detect_renovations", "Record detected major upgrades.",
            {"renovations": {"type": "array", "items": {"type": "string"}}}),
    fn_spec("extract_site_features", "Record site features (frontage, corner lot, backing, view, cul-de-sac).",
            {"features": {"type": "object"}}),
    fn_spec("check_measurement_conflicts", "Compare listing GLA/lot vs assessment/survey."),
    fn_spec("validate_subject_schema", "Finalize coordinates/age and run the subject data-quality gate."),
] + SHARED_TOOL_SPECS


def build_dispatch(tk: SubjectToolkit) -> dict[str, Any]:
    d = {
        "extract_subject_from_listing": tk.extract_subject_from_listing,
        "inspect_photos": tk.inspect_photos,
        "parse_floor_plan": tk.parse_floor_plan,
        "normalize_measurement": tk.normalize_measurement,
        "classify_property_type": tk.classify_property_type,
        "classify_condition": tk.classify_condition,
        "classify_quality": tk.classify_quality,
        "detect_renovations": tk.detect_renovations,
        "extract_site_features": tk.extract_site_features,
        "check_measurement_conflicts": tk.check_measurement_conflicts,
        "validate_subject_schema": tk.validate_subject_schema,
    }
    d.update(tk.shared_dispatch())
    return d
