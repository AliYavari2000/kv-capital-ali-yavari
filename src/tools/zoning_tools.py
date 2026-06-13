"""Tools for Node 4 - Zoning, Land-Use, Legality, Highest-and-Best-Use (LLM agent).

The HBU four-test logic and the permitted-use checks are rule-based; the agent
orchestrates the lookups and writes the explanation. GIS / bylaw documents are
read from the case where structured (geojson / notes) and exposed as hooks
otherwise.
"""

from __future__ import annotations

from typing import Any, Optional

from src import case_store
from src.tools.base import SHARED_TOOL_SPECS, ToolkitBase, fn_spec

_ZONING_BY_TYPE = {
    "Detached": {"code": "R-C1 / RF1", "permitted": "Single detached dwelling"},
    "Semi-Detached": {"code": "R-C2 / RF3", "permitted": "Semi-detached / duplex dwelling"},
    "Townhouse": {"code": "R-CG / RF5", "permitted": "Row / townhouse dwelling"},
    "Condo": {"code": "M-C1 / RA7", "permitted": "Multi-residential (apartment)"},
}


class ZoningToolkit(ToolkitBase):
    node_name = "zoning_hbu"

    def __init__(self, state: dict[str, Any]) -> None:
        super().__init__()
        self.case_dir = state.get("case_dir")
        s = state.get("subject", {})
        self.ptype = s.get("property_type")
        self.lot = s.get("lot_size_sqft")
        self.record: dict[str, Any] = {}
        self.hbu_tests: dict[str, bool] = {}
        self._documents = dict(state.get("documents", {}))

    def municipal_zoning_lookup(self) -> dict[str, Any]:
        """Get the zoning district for the subject's property type."""
        z = _ZONING_BY_TYPE.get(self.ptype)
        if z:
            self.record.update({"zoning_code": z["code"], "permitted_use": z["permitted"]})
            return {"found": True, **z}
        self.record["zoning_code"] = "Unverified"
        return {"found": False, "note": "property type unknown; zoning not determined"}

    def zoning_map_query(self) -> dict[str, Any]:
        """Query the GIS land-use district layer from the case."""
        if not self.case_dir:
            return {"found": False}
        lud = case_store.load_case(self.case_dir).zoning()["data"].get("land_use_district")
        if isinstance(lud, dict) and lud.get("features"):
            return {"found": True, "properties": lud["features"][0].get("properties", {})}
        return {"found": False}

    def bylaw_document_search(self, query: str = "") -> dict[str, Any]:
        """Search the zoning bylaw text (document hook; extraction pending)."""
        doc = self._documents.get("zoning", {}).get("zoning_bylaw_excerpt", {})
        return {"bylaw_present": bool(doc.get("exists")), "parsed": False, "query": query}

    def permitted_use_checker(self) -> dict[str, Any]:
        """Check whether the current use is permitted/discretionary/prohibited."""
        permitted = self.ptype in _ZONING_BY_TYPE
        status = "permitted" if permitted else "unverified"
        self.record["use_status"] = status
        return {"use_status": status, "current_use": self.ptype}

    def development_standard_extractor(self) -> dict[str, Any]:
        """Extract setbacks/height/coverage/density/parking (representative defaults)."""
        std = {"front_setback_m": 6.0, "side_setback_m": 1.2, "max_height_m": 10.0,
               "max_lot_coverage": 0.45, "min_parking": 2}
        self.record["development_standards"] = std
        return std

    def nonconforming_use_detector(self) -> dict[str, Any]:
        """Flag potential legal nonconforming use (rule-based)."""
        nonconforming = False
        reason = None
        if self.ptype == "Detached" and self.lot is not None and float(self.lot) < 3000:
            nonconforming = True
            reason = f"Detached use on a {int(float(self.lot))} sqft lot may be legal nonconforming."
        self.record["nonconforming"] = nonconforming
        return {"nonconforming": nonconforming, "reason": reason}

    def overlay_lookup(self) -> dict[str, Any]:
        """Check floodplain/heritage/airport/environmental/redevelopment overlays."""
        if not self.case_dir:
            return {"overlays": []}
        ov = case_store.load_case(self.case_dir).zoning()["data"].get("overlays")
        overlays = []
        if isinstance(ov, dict) and ov.get("features"):
            for f in ov["features"]:
                o = f.get("properties", {}).get("overlay")
                if o and o != "none":
                    overlays.append(o)
        self.record["overlays"] = overlays
        return {"overlays": overlays}

    def permit_history_lookup(self) -> dict[str, Any]:
        """Look for building/development permits in the case."""
        if not self.case_dir:
            return {"building_permits": 0, "development_permits": 0}
        perms = case_store.load_case(self.case_dir).zoning()["data"].get("permits", {})
        b = perms.get("building_permits") or []
        d = perms.get("development_permits") or []
        out = {"building_permits": len(b) if isinstance(b, list) else 0,
               "development_permits": len(d) if isinstance(d, list) else 0}
        self.record["permits"] = out
        return out

    def hbu_rule_engine(self, legally_permissible: bool, physically_possible: bool,
                        financially_feasible: bool, maximally_productive: bool) -> dict[str, Any]:
        """Apply the four HBU tests (the agent supplies each boolean conclusion)."""
        tests = {
            "legally_permissible": bool(legally_permissible),
            "physically_possible": bool(physically_possible),
            "financially_feasible": bool(financially_feasible),
            "maximally_productive": bool(maximally_productive),
        }
        self.hbu_tests = tests
        passes = all(tests.values())
        self.record["hbu_as_improved"] = passes
        return {"tests": tests, "present_use_is_hbu": passes}

    def zoning_summary_generator(self, summary: str) -> dict[str, Any]:
        """Store the agent's concise zoning/HBU explanation."""
        self.record["summary"] = summary
        return {"stored": True}


TOOL_SPECS = [
    fn_spec("municipal_zoning_lookup", "Get the zoning district for the subject."),
    fn_spec("zoning_map_query", "Query the GIS land-use district layer."),
    fn_spec("bylaw_document_search", "Search the zoning bylaw text (extraction pending).",
            {"query": {"type": "string"}}),
    fn_spec("permitted_use_checker", "Check whether the current use is permitted/discretionary/prohibited."),
    fn_spec("development_standard_extractor", "Extract setbacks/height/coverage/density/parking."),
    fn_spec("nonconforming_use_detector", "Flag potential legal nonconforming use."),
    fn_spec("overlay_lookup", "Check floodplain/heritage/airport/environmental/redevelopment overlays."),
    fn_spec("permit_history_lookup", "Look for building/development permits."),
    fn_spec("hbu_rule_engine",
            "Apply the four highest-and-best-use tests. Provide your boolean conclusion for each.",
            {"legally_permissible": {"type": "boolean"},
             "physically_possible": {"type": "boolean"},
             "financially_feasible": {"type": "boolean"},
             "maximally_productive": {"type": "boolean"}},
            ["legally_permissible", "physically_possible", "financially_feasible", "maximally_productive"]),
    fn_spec("zoning_summary_generator", "Store a concise zoning/HBU explanation.",
            {"summary": {"type": "string"}}, ["summary"]),
] + SHARED_TOOL_SPECS


def build_dispatch(tk: ZoningToolkit) -> dict[str, Any]:
    d = {
        "municipal_zoning_lookup": tk.municipal_zoning_lookup,
        "zoning_map_query": tk.zoning_map_query,
        "bylaw_document_search": tk.bylaw_document_search,
        "permitted_use_checker": tk.permitted_use_checker,
        "development_standard_extractor": tk.development_standard_extractor,
        "nonconforming_use_detector": tk.nonconforming_use_detector,
        "overlay_lookup": tk.overlay_lookup,
        "permit_history_lookup": tk.permit_history_lookup,
        "hbu_rule_engine": tk.hbu_rule_engine,
        "zoning_summary_generator": tk.zoning_summary_generator,
    }
    d.update(tk.shared_dispatch())
    return d
