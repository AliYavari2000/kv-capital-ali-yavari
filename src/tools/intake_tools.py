"""Tools for Node 1 - Assignment Intake (LLM agent).

The intake node is an LLM agent: the model decides which of these tools to call,
in what order, to define the assignment (type, effective date, intended use,
client requirements), build the subject's preliminary fact set, determine the
required-document checklist, flag missing information, mint a workfile ID, log
evidence, and escalate ambiguous assignments to a human.

The tools themselves do deterministic work (reading the structured inputs,
validating, building checklists). The *agentic* judgment -- classifying the
assignment type, deciding what's ambiguous, choosing what to escalate -- is the
LLM's. There is no non-LLM fallback for the node; see ``node1_assignment_intake``.

A single :class:`IntakeToolkit` instance is bound per node invocation; its
methods are exposed to the model and accumulate results the node reads back out.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from typing import Any, Optional

from src import case_store, config, llm

# Canonical preliminary subject fields the intake produces.
SUBJECT_FIELDS = [
    "address", "city", "neighborhood", "property_type",
    "bedrooms", "bathrooms", "gla_sqft", "lot_size_sqft", "year_built",
]

ASSIGNMENT_TYPES = [
    "purchase", "refinance", "construction_loan",
    "retrospective", "estate", "other",
]

# Documents a residential appraisal assignment generally requires.
_BASE_REQUIRED_DOCS = [
    "listing", "title", "rpr", "permits", "tax_assessment", "photos",
]
_TYPE_EXTRA_DOCS = {
    "purchase": ["purchase_contract"],
    "construction_loan": ["building_permits", "plans_and_specs"],
    "retrospective": ["historical_title_search"],
}


class IntakeToolkit:
    """Per-invocation context + tools for the assignment-intake agent."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.raw_input = state.get("raw_input")
        self.case_dir = state.get("case_dir")

        # Resolve the structured inputs available to this assignment.
        self._assignment_data: dict[str, Any] = {}
        self._assignment_docs: dict[str, Any] = {}
        self._subject_docs: dict[str, Any] = {}
        self._listing_rows: list[dict[str, Any]] = []
        self.documents: dict[str, Any] = dict(state.get("documents", {}))

        if self.case_dir:
            case = case_store.load_case(self.case_dir)
            self.documents = case.document_index()
            assignment_sec = case.assignment()
            subject_sec = case.subject()
            eff = assignment_sec["data"].get("effective_date") or {}
            self._assignment_data = dict(eff) if isinstance(eff, dict) else {}
            self._assignment_docs = assignment_sec["documents"]
            self._subject_docs = subject_sec["documents"]
            listing = subject_sec["data"].get("listing") or []
            if isinstance(listing, list):
                self._listing_rows = case_store.normalize_comp_columns(listing)
        elif isinstance(self.raw_input, dict):
            self._assignment_data = dict(self.raw_input)
            self._listing_rows = [dict(self.raw_input)]

        # Accumulated results, read back by the node after the agent loop.
        self.assignment_request: dict[str, Any] = {}
        self.listing: dict[str, Any] = {}
        self.instructions: dict[str, Any] = {}
        self.effective_date_check: dict[str, Any] = {}
        self.assignment_type: dict[str, Any] = {}
        self.checklist: dict[str, Any] = {}
        self.missing_info: dict[str, Any] = {}
        self.workfile_id: Optional[str] = None
        self.evidence: list[dict[str, Any]] = []
        self.escalations: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ tools
    def parse_assignment_request(self) -> dict[str, Any]:
        """Extract assignment type hint, borrower/client, subject address,
        valuation purpose, and effective date from the assignment request."""
        d = self._assignment_data
        out = {
            "client": d.get("client"),
            "borrower": d.get("borrower"),
            "subject_address": d.get("address") or d.get("subject_address"),
            "valuation_purpose": d.get("intended_use") or d.get("valuation_purpose"),
            "effective_date": d.get("effective_date"),
            "report_date": d.get("report_date"),
            "assignment_type_hint": d.get("assignment_type"),
            "source": "case_folder" if self.case_dir else ("structured" if d else "free_text"),
        }
        self.assignment_request = out
        return out

    def parse_listing_input(self) -> dict[str, Any]:
        """Read MLS/listing input and extract structured property details."""
        fields: dict[str, Any] = {}
        if self._listing_rows:
            row = self._listing_rows[0]
            fields = {k: row.get(k) for k in SUBJECT_FIELDS if row.get(k) not in (None, "")}
        elif isinstance(self.raw_input, str) and self.raw_input.strip():
            # Messy free-text listing -> structured fields via the LLM extractor.
            parsed = llm.parse_listing(self.raw_input)
            fields = {k: parsed.get(k) for k in SUBJECT_FIELDS if parsed.get(k) not in (None, "")}
        self.listing = fields
        return {"property_details": fields, "fields_found": sorted(fields.keys())}

    def parse_client_instructions(self) -> dict[str, Any]:
        """Extract special requirements from lender/client instructions."""
        d = self._assignment_data
        reqs = d.get("special_requirements") or d.get("instructions") or []
        if isinstance(reqs, str):
            reqs = [reqs]
        instr_doc = self._assignment_docs.get("lender_instructions")
        out = {
            "special_requirements": list(reqs),
            "instructions_document_present": bool(instr_doc and instr_doc.get("exists")),
            "instructions_document_parsed": bool(instr_doc and instr_doc.get("parsed")),
        }
        if out["instructions_document_present"] and not out["instructions_document_parsed"]:
            out["note"] = "lender_instructions.pdf present but awaiting an extraction tool."
        self.instructions = out
        return out

    def validate_effective_date(self, effective_date: Optional[str] = None) -> dict[str, Any]:
        """Check whether the effective date is present and logically valid."""
        value = effective_date or self.assignment_request.get("effective_date")
        issues: list[str] = []
        normalized = None
        present = bool(value)
        if not present:
            issues.append("effective date is missing")
        else:
            try:
                parsed = _dt.date.fromisoformat(str(value)[:10])
                normalized = parsed.isoformat()
                if parsed.year < 1990:
                    issues.append("effective date is implausibly old")
                if parsed > _dt.date.today() + _dt.timedelta(days=1):
                    issues.append("effective date is in the future")
            except ValueError:
                issues.append(f"effective date '{value}' is not a valid ISO date")
        out = {"present": present, "valid": not issues, "normalized": normalized, "issues": issues}
        self.effective_date_check = out
        return out

    def detect_assignment_type(self, assignment_type: str, rationale: str = "") -> dict[str, Any]:
        """Record the assignment type the agent concluded (purchase, refinance,
        construction_loan, retrospective, estate, other), with rationale."""
        chosen = (assignment_type or "").strip().lower().replace(" ", "_")
        valid = chosen in ASSIGNMENT_TYPES
        if not valid:
            chosen = "other"
        out = {"assignment_type": chosen, "rationale": rationale, "valid": valid,
               "allowed": ASSIGNMENT_TYPES}
        self.assignment_type = out
        return out

    def required_document_checklist(self, assignment_type: Optional[str] = None) -> dict[str, Any]:
        """Determine required documents for the assignment and which are present."""
        atype = (assignment_type or self.assignment_type.get("assignment_type") or "other")
        required = list(_BASE_REQUIRED_DOCS) + _TYPE_EXTRA_DOCS.get(atype, [])

        present = self._present_document_kinds()
        # Map checklist items to the document evidence actually on file.
        present_items, missing_items = [], []
        for item in required:
            if any(item in key or key in item for key in present):
                present_items.append(item)
            else:
                missing_items.append(item)
        out = {"assignment_type": atype, "required": required,
               "present": present_items, "missing": missing_items}
        self.checklist = out
        return out

    def missing_info_detector(self) -> dict[str, Any]:
        """Flag missing critical assignment/subject information."""
        missing: list[str] = []
        ar = self.assignment_request
        subj = self.listing
        if not (ar.get("subject_address") or subj.get("address")):
            missing.append("subject_address")
        if not subj.get("property_type"):
            missing.append("property_type")
        if not self.effective_date_check.get("present", bool(ar.get("effective_date"))):
            missing.append("effective_date")
        # Legal description is not part of intake's structured inputs yet.
        if not self._assignment_data.get("legal_description"):
            missing.append("legal_description")
        out = {"missing": missing, "complete": not missing}
        self.missing_info = out
        return out

    def create_workfile_id(self) -> dict[str, Any]:
        """Create a unique valuation/workfile ID."""
        seed = "|".join(str(x) for x in (
            self.case_dir,
            self.assignment_request.get("subject_address") or self.listing.get("address"),
            self.assignment_request.get("borrower"),
            self.assignment_request.get("effective_date"),
        ))
        suffix = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:6].upper()
        wid = f"KV-{config.VALUATION_DATE:%Y%m%d}-{suffix}"
        self.workfile_id = wid
        return {"workfile_id": wid}

    def append_evidence(self, source: str, detail: str = "") -> dict[str, Any]:
        """Log an assignment source document / fact into the evidence trail."""
        entry = {
            "node": "assignment_intake",
            "source": source,
            "detail": detail,
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        self.evidence.append(entry)
        return {"logged": True, "evidence_count": len(self.evidence)}

    def raise_human_review(self, reason: str, severity: str = "medium") -> dict[str, Any]:
        """Escalate an ambiguous/incomplete assignment for human review."""
        sev = severity if severity in ("low", "medium", "high") else "medium"
        self.escalations.append({"reason": reason, "severity": sev})
        return {"escalated": True, "escalation_count": len(self.escalations)}

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _any_exists(values: Any) -> bool:
        """True if any value (a hook dict, or a list of hook dicts) exists."""
        for v in values:
            if isinstance(v, dict) and v.get("exists"):
                return True
            if isinstance(v, list) and any(
                    isinstance(x, dict) and x.get("exists") for x in v):
                return True
        return False

    def _present_document_kinds(self) -> set[str]:
        """Set of document keys present (exists=True) across assignment/subject
        sections, plus a synthetic 'listing' when listing data was read."""
        present: set[str] = set()
        if self.listing or self._listing_rows:
            present.add("listing")

        def _scan(docs: dict[str, Any]) -> None:
            for key, val in docs.items():
                if isinstance(val, dict) and val.get("exists"):
                    present.add(key)
                elif isinstance(val, list):
                    if any(isinstance(v, dict) and v.get("exists") for v in val):
                        present.add(key)

        _scan(self._assignment_docs)
        _scan(self._subject_docs)
        # Title / RPR / tax / permits live in other sections; surface them if the
        # case exposes them so the checklist reflects the whole workfile.
        if self.case_dir:
            try:
                case = case_store.load_case(self.case_dir)
                _scan(case.legal_title()["documents"])
                _scan(case.zoning()["documents"])
                # Normalize a few names to checklist vocabulary.
                lt_docs = case.legal_title()["documents"]
                if any(isinstance(v, dict) and v.get("exists") for k, v in lt_docs.items()
                       if "title" in k):
                    present.add("title")
                if lt_docs.get("real_property_report", {}).get("exists"):
                    present.add("rpr")
                assess = lt_docs.get("assessment", {})
                if isinstance(assess, dict) and any(
                        isinstance(v, dict) and v.get("exists") for v in assess.values()):
                    present.add("tax_assessment")
                z_docs = case.zoning()["documents"]
                perms = z_docs.get("permits", {})
                if isinstance(perms, dict) and self._any_exists(perms.values()):
                    present.add("permits")
                if any(isinstance(v, dict) and v.get("exists")
                       for v in self._subject_docs.get("subject_photos", []) or []):
                    present.add("photos")
            except Exception:
                pass
        return present


# OpenAI tool (function) schemas the agent loop binds.
TOOL_SPECS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "parse_assignment_request",
        "description": "Extract assignment type hint, borrower/client, subject address, "
                       "valuation purpose, and effective date from the assignment request.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "parse_listing_input",
        "description": "Read MLS/listing input and extract structured property details "
                       "(address, type, beds/baths, GLA, lot, year built).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "parse_client_instructions",
        "description": "Extract special requirements from lender/client instructions.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "validate_effective_date",
        "description": "Check whether the effective (valuation) date is present and logically valid.",
        "parameters": {"type": "object", "properties": {
            "effective_date": {"type": "string",
                               "description": "ISO date to validate; omit to use the one already parsed."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "detect_assignment_type",
        "description": "Record the assignment type you concluded. One of: "
                       + ", ".join(ASSIGNMENT_TYPES) + ".",
        "parameters": {"type": "object", "properties": {
            "assignment_type": {"type": "string", "enum": ASSIGNMENT_TYPES},
            "rationale": {"type": "string"}},
            "required": ["assignment_type"]}}},
    {"type": "function", "function": {
        "name": "required_document_checklist",
        "description": "Determine the required-document checklist for the assignment and which "
                       "documents are present in the workfile.",
        "parameters": {"type": "object", "properties": {
            "assignment_type": {"type": "string", "enum": ASSIGNMENT_TYPES}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "missing_info_detector",
        "description": "Flag missing critical info (address, property type, effective date, "
                       "legal description).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "create_workfile_id",
        "description": "Create a unique valuation/workfile ID for this assignment.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "append_evidence",
        "description": "Log an assignment source document or fact into the evidence trail.",
        "parameters": {"type": "object", "properties": {
            "source": {"type": "string"}, "detail": {"type": "string"}},
            "required": ["source"]}}},
    {"type": "function", "function": {
        "name": "raise_human_review",
        "description": "Escalate an ambiguous or incomplete assignment for human review.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]}},
            "required": ["reason"]}}},
]


def build_dispatch(toolkit: IntakeToolkit) -> dict[str, Any]:
    """Map tool names -> bound toolkit methods for the agent loop."""
    return {
        "parse_assignment_request": toolkit.parse_assignment_request,
        "parse_listing_input": toolkit.parse_listing_input,
        "parse_client_instructions": toolkit.parse_client_instructions,
        "validate_effective_date": toolkit.validate_effective_date,
        "detect_assignment_type": toolkit.detect_assignment_type,
        "required_document_checklist": toolkit.required_document_checklist,
        "missing_info_detector": toolkit.missing_info_detector,
        "create_workfile_id": toolkit.create_workfile_id,
        "append_evidence": toolkit.append_evidence,
        "raise_human_review": toolkit.raise_human_review,
    }
