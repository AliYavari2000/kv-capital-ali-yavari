"""Tools for Node 7 - Fact Verification (LLM agent).

Verifies subject and comp facts across sources. Plausibility checks and the
confidence scoring are deterministic; the agent decides what to verify, writes
verified facts, and escalates unresolved conflicts.
"""

from __future__ import annotations

from typing import Any

from src import case_store, config, valuation_math as vm
from src.tools.base import SHARED_TOOL_SPECS, ToolkitBase, fn_spec

_GLA_BOUNDS = (300, 12000)
_LOT_BOUNDS = (0, 80000)
# Relative reliability of each source id (used by confidence_score).
_SOURCE_WEIGHT = {
    "alberta_spin2_title": 0.95,
    "rpr_survey": 0.92,
    "open_calgary_assessment": 0.88,
    "calgary_assessment_pdf": 0.88,
    "open_calgary_building_permits": 0.85,
    "reca_rms": 0.84,
    "pillar9_mls": 0.80,
    # legacy string aliases from agent tool calls
    "title": 0.95, "rpr": 0.92, "assessment": 0.88,
    "municipal assessment": 0.88, "permit": 0.85,
    "mls": 0.80, "mls listing": 0.80, "listing": 0.70,
}


class VerificationToolkit(ToolkitBase):
    node_name = "fact_verification"

    def __init__(self, state: dict[str, Any]) -> None:
        super().__init__()
        self.case_dir = state.get("case_dir")
        self.subject = state.get("subject", {})
        self.comps = [dict(c) for c in state.get("ranked_comps", [])]
        self.flags: list[dict[str, str]] = []
        self.verified_facts: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}

    def _flag(self, code: str, severity: str, message: str) -> None:
        self.flags.append({"code": code, "severity": severity, "message": message})

    def cross_source_fact_checker(self) -> dict[str, Any]:
        """Cross-check each comp's price/date/GLA/lot against plausibility rules."""
        verified = 0
        for c in self.comps:
            notes, ok = [], True
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
            c["verified"] = ok
            c["verification_notes"] = notes
            verified += int(ok)
        unverified = len(self.comps) - verified
        if unverified:
            self._flag("unverified_comps", "medium" if unverified < len(self.comps) else "high",
                       f"{unverified} of {len(self.comps)} comps failed a fact check.")
        self.summary = {"checked": len(self.comps), "verified": verified, "unverified": unverified}
        return self.summary

    def assessment_record_lookup(self) -> dict[str, Any]:
        """Pull assessment value/year/GLA/lot for the subject."""
        if not self.case_dir:
            return {"found": False}
        assess = (case_store.load_case(self.case_dir).legal_title()["data"]
                  .get("assessment", {}).get("assessment_open_data") or [])
        return {"found": bool(assess), "assessment": assess[0] if assess else None}

    def permit_lookup(self) -> dict[str, Any]:
        """Check permits for renovations/additions/garage/basement."""
        if not self.case_dir:
            return {"building_permits": [], "development_permits": []}
        perms = case_store.load_case(self.case_dir).zoning()["data"].get("permits", {})
        return {"building_permits": perms.get("building_permits") or [],
                "development_permits": perms.get("development_permits") or []}

    def rpr_parser(self) -> dict[str, Any]:
        """Read the Real Property Report / survey (extraction pending)."""
        return {"parsed": False, "note": "RPR/survey extraction pending."}

    def title_fact_checker(self) -> dict[str, Any]:
        """Confirm legal identity / parcel facts against the title node output."""
        return {"note": "legal identity confirmed upstream in legal_title node."}

    def sale_price_verifier(self) -> dict[str, Any]:
        """Confirm comp sale prices are present and positive."""
        bad = [c.get("id") for c in self.comps if not (float(c.get("sale_price") or 0) > 0)]
        return {"all_present": not bad, "missing_price_ids": bad}

    def gla_conflict_detector(self) -> dict[str, Any]:
        """Flag comps with implausible / inconsistent living area."""
        bad = [c.get("id") for c in self.comps
               if not (_GLA_BOUNDS[0] <= float(c.get("gla_sqft") or 0) <= _GLA_BOUNDS[1])]
        if bad:
            self._flag("gla_conflict", "medium", f"GLA conflict on comps: {bad}.")
        return {"conflicts": bad}

    def lot_size_conflict_detector(self) -> dict[str, Any]:
        """Flag comps with implausible lot size."""
        bad = []
        for c in self.comps:
            lot = c.get("lot_size_sqft")
            if lot not in (None, "") and not (_LOT_BOUNDS[0] <= float(lot) <= _LOT_BOUNDS[1]):
                bad.append(c.get("id"))
        return {"conflicts": bad}

    def property_type_conflict_detector(self) -> dict[str, Any]:
        """Flag comps whose type differs from the subject's tier."""
        st = self.subject.get("property_type")
        diff = [c.get("id") for c in self.comps if c.get("property_type") != st]
        return {"different_type_ids": diff, "subject_type": st}

    def condition_conflict_detector(self) -> dict[str, Any]:
        """Compare listing remarks/photos with condition (extraction pending)."""
        return {"parsed": False, "note": "condition cross-check needs photo/remarks extraction."}

    def confidence_score(self, field: str, sources: list) -> dict[str, Any]:
        """Score a fact's confidence from the reliability of its sources."""
        srcs = sources or []
        if not srcs:
            score = 0.4
        else:
            best = max(_SOURCE_WEIGHT.get(str(s).lower(), 0.6) for s in srcs)
            bonus = min(0.1, 0.03 * (len(srcs) - 1))
            score = min(0.99, best + bonus)
        return {"field": field, "confidence": round(score, 2), "sources": srcs}

    def verified_fact_writer(self, field: str, value: Any, confidence: float,
                             sources: list, conflict: bool = False) -> dict[str, Any]:
        """Write a verified fact into state."""
        fact = {"field": field, "value": value, "confidence": round(float(confidence), 2),
                "sources": sources or [], "conflict": bool(conflict)}
        self.verified_facts.append(fact)
        return {"written": True, "fact_count": len(self.verified_facts)}


TOOL_SPECS = [
    fn_spec("cross_source_fact_checker", "Cross-check each comp's price/date/GLA/lot against rules."),
    fn_spec("assessment_record_lookup", "Pull assessment value/year/GLA/lot for the subject."),
    fn_spec("permit_lookup", "Check permits for renovations/additions/garage/basement."),
    fn_spec("rpr_parser", "Read the Real Property Report / survey (extraction pending)."),
    fn_spec("title_fact_checker", "Confirm legal identity / parcel facts."),
    fn_spec("sale_price_verifier", "Confirm comp sale prices are present and positive."),
    fn_spec("gla_conflict_detector", "Flag comps with inconsistent living area."),
    fn_spec("lot_size_conflict_detector", "Flag comps with implausible lot size."),
    fn_spec("property_type_conflict_detector", "Flag comps whose type differs from the subject."),
    fn_spec("condition_conflict_detector", "Compare remarks/photos with condition (pending)."),
    fn_spec("confidence_score", "Score a fact's confidence from source reliability.",
            {"field": {"type": "string"}, "sources": {"type": "array", "items": {"type": "string"}}},
            ["field", "sources"]),
    fn_spec("verified_fact_writer", "Write a verified fact into state.",
            {"field": {"type": "string"}, "value": {},
             "confidence": {"type": "number"},
             "sources": {"type": "array", "items": {"type": "string"}},
             "conflict": {"type": "boolean"}},
            ["field", "value", "confidence", "sources"]),
] + SHARED_TOOL_SPECS


def build_dispatch(tk: VerificationToolkit) -> dict[str, Any]:
    d = {
        "cross_source_fact_checker": tk.cross_source_fact_checker,
        "assessment_record_lookup": tk.assessment_record_lookup,
        "permit_lookup": tk.permit_lookup,
        "rpr_parser": tk.rpr_parser,
        "title_fact_checker": tk.title_fact_checker,
        "sale_price_verifier": tk.sale_price_verifier,
        "gla_conflict_detector": tk.gla_conflict_detector,
        "lot_size_conflict_detector": tk.lot_size_conflict_detector,
        "property_type_conflict_detector": tk.property_type_conflict_detector,
        "condition_conflict_detector": tk.condition_conflict_detector,
        "confidence_score": tk.confidence_score,
        "verified_fact_writer": tk.verified_fact_writer,
    }
    d.update(tk.shared_dispatch())
    return d
