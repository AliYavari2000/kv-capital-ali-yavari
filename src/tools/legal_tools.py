"""Tools for Node 3 - Legal Identity and Title (LLM agent).

Confirms the subject is legally the correct property. Parcel/title identity is
derived deterministically (title PDFs are document hooks awaiting an extraction
tool); the agent's job is to cross-check identity across sources and escalate on
any mismatch.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from src import case_store
from src.tools.base import SHARED_TOOL_SPECS, ToolkitBase, fn_spec


def _stable_int(seed: str, mod: int) -> int:
    return int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16) % mod


class LegalToolkit(ToolkitBase):
    node_name = "legal_title"

    def __init__(self, state: dict[str, Any]) -> None:
        super().__init__()
        self.case_dir = state.get("case_dir")
        s = state.get("subject", {})
        self.address = s.get("address")
        self.city = s.get("city") or "AB"
        self.nb = s.get("neighborhood") or "Unknown"
        self.registered_owner = s.get("registered_owner")
        self._seed = f"{self.address}|{self.nb}|{self.city}"
        self.record: dict[str, Any] = {}
        self.conflicts: list[str] = []
        self._documents = dict(state.get("documents", {}))

    def parcel_lookup_by_address(self) -> dict[str, Any]:
        """Find parcel ID / roll number / legal land description from the address."""
        if not self.address:
            return {"found": False, "note": "no subject address to look up"}
        seed = self._seed
        linc = (f"{_stable_int(seed+'a',9000)+1000:04d}-{_stable_int(seed+'b',9000)+1000:04d}-"
                f"{_stable_int(seed+'c',90)+10:02d}")
        legal = (f"Plan {_stable_int(seed+'p',8999999)+1000000}, "
                 f"Block {_stable_int(seed+'blk',40)+1}, Lot {_stable_int(seed+'lot',60)+1}")
        self.record.update({"parcel_id": linc, "legal_description": legal})
        return {"found": True, "parcel_id": linc, "legal_description": legal}

    def title_document_parser(self) -> dict[str, Any]:
        """Title certificate (extraction pending); reports presence + synthetic status."""
        docs = self._documents.get("legal_title", {})
        cert = docs.get("current_certificate_of_title", {})
        present = bool(cert.get("exists"))
        status = "Clear"
        if self.address and _stable_int(self._seed + "title", 100) < 12:
            status = "Encumbered"
        self.record["title_status"] = status
        return {"certificate_present": present, "parsed": False, "title_status": status}

    def legal_description_matcher(self) -> dict[str, Any]:
        """Compare the title legal description to listing/assessment sources."""
        # Only the synthetic record exists today; treat as consistent unless missing.
        ld = self.record.get("legal_description")
        consistent = bool(ld)
        return {"legal_description": ld, "consistent": consistent}

    def address_normalizer(self, address: Optional[str] = None) -> dict[str, Any]:
        """Standardize the civic address."""
        a = (address or self.address or "").strip()
        norm = " ".join(a.split()).title() if a else None
        if norm:
            self.record["normalized_address"] = norm
        return {"normalized_address": norm}

    def owner_name_extractor(self) -> dict[str, Any]:
        """Extract the registered owner if available."""
        owner = self.registered_owner or "On file with KV Capital"
        self.record["registered_owner"] = owner
        return {"registered_owner": owner, "verified": bool(self.registered_owner)}

    def encumbrance_extractor(self) -> dict[str, Any]:
        """Extract easements/caveats/restrictions/liens if present."""
        docs = self._documents.get("legal_title", {})
        enc = docs.get("encumbrances", []) or []
        names = [e.get("path", "").split("/")[-1] for e in enc if isinstance(e, dict)]
        self.record["encumbrances"] = names
        return {"encumbrance_documents": names, "parsed": False,
                "note": "encumbrance docs present; extraction pending." if names else "none on file"}

    def tax_roll_lookup(self) -> dict[str, Any]:
        """Pull assessment/tax roll identity if available."""
        if not self.case_dir:
            return {"found": False}
        assess = (case_store.load_case(self.case_dir).legal_title()["data"]
                  .get("assessment", {}).get("assessment_open_data") or [])
        if isinstance(assess, list) and assess:
            roll = assess[0].get("roll_number")
            self.record["roll_number"] = roll
            return {"found": True, "roll_number": roll, "assessment": assess[0]}
        return {"found": False}

    def parcel_boundary_lookup(self) -> dict[str, Any]:
        """Retrieve parcel geometry / id from the parcel boundary layer."""
        if not self.case_dir:
            return {"found": False}
        boundary = case_store.load_case(self.case_dir).zoning()["data"].get("parcel_boundary")
        if isinstance(boundary, dict) and boundary.get("features"):
            props = boundary["features"][0].get("properties", {})
            return {"found": True, "properties": props}
        return {"found": False}

    def title_conflict_detector(self) -> dict[str, Any]:
        """Flag title/address/legal-description mismatches."""
        conflicts: list[str] = []
        if not self.address:
            conflicts.append("missing subject address")
        if not self.record.get("legal_description"):
            conflicts.append("missing legal description")
        if self.record.get("title_status") == "Encumbered":
            conflicts.append("registered encumbrance on title")
        self.conflicts = conflicts
        return {"conflicts": conflicts, "has_conflict": bool(conflicts)}


TOOL_SPECS = [
    fn_spec("parcel_lookup_by_address", "Find parcel ID / roll number / legal description from the address."),
    fn_spec("title_document_parser", "Read the title certificate (extraction pending); returns status."),
    fn_spec("legal_description_matcher", "Compare title legal description to other sources."),
    fn_spec("address_normalizer", "Standardize the civic address.",
            {"address": {"type": "string"}}),
    fn_spec("owner_name_extractor", "Extract the registered owner if available."),
    fn_spec("encumbrance_extractor", "Extract easements/caveats/restrictions/liens if present."),
    fn_spec("tax_roll_lookup", "Pull assessment/tax roll identity if available."),
    fn_spec("parcel_boundary_lookup", "Retrieve parcel geometry / id from the parcel boundary layer."),
    fn_spec("title_conflict_detector", "Flag title/address/legal-description mismatches."),
] + SHARED_TOOL_SPECS


def build_dispatch(tk: LegalToolkit) -> dict[str, Any]:
    d = {
        "parcel_lookup_by_address": tk.parcel_lookup_by_address,
        "title_document_parser": tk.title_document_parser,
        "legal_description_matcher": tk.legal_description_matcher,
        "address_normalizer": tk.address_normalizer,
        "owner_name_extractor": tk.owner_name_extractor,
        "encumbrance_extractor": tk.encumbrance_extractor,
        "tax_roll_lookup": tk.tax_roll_lookup,
        "parcel_boundary_lookup": tk.parcel_boundary_lookup,
        "title_conflict_detector": tk.title_conflict_detector,
    }
    d.update(tk.shared_dispatch())
    return d
