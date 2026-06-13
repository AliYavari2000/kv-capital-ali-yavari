"""Node 3 - Legal Identity and Title (LLM agent).

Confirms the subject being analyzed is legally the correct property: parcel /
roll identity, legal description, registered owner, encumbrances, and any
title/address/legal-description conflicts. Title certificates are document hooks
awaiting extraction; the agent cross-checks identity and escalates on mismatch.

No deterministic fallback: requires an LLM.
"""

from __future__ import annotations

from typing import Any

from src import llm
from src.state import CompState
from src.tools import legal_tools

_SYSTEM_PROMPT = (
    "You are the Legal Identity and Title agent for KV Capital. Confirm the "
    "subject property's legal identity using ONLY the tools. Do not invent owners, "
    "parcels, or legal descriptions.\n\n"
    "Steps: parcel_lookup_by_address; address_normalizer; title_document_parser; "
    "tax_roll_lookup and parcel_boundary_lookup to corroborate identity; "
    "owner_name_extractor; encumbrance_extractor; legal_description_matcher; then "
    "title_conflict_detector. append_evidence for each source. You MUST call "
    "raise_human_review if the title identity does not match the subject "
    "(address mismatch, legal-description mismatch, multiple parcels, condo unit "
    "ambiguity, unclear ownership, value-affecting easement/restriction, or "
    "missing title). Finish with a one-sentence identity confirmation."
)


def _flag(code: str, severity: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def legal_title_node(state: CompState) -> dict[str, Any]:
    if not llm.llm_available():
        raise llm.LLMUnavailableError(
            "Legal/Title is an LLM agent and requires OPENAI_API_KEY to be set."
        )

    tk = legal_tools.LegalToolkit(state)
    dispatch = legal_tools.build_dispatch(tk)
    import json
    user = ("Confirm the legal identity of this subject:\n"
            + json.dumps({"address": tk.address, "neighborhood": tk.nb, "city": tk.city}, indent=2))
    run = llm.run_tool_agent(_SYSTEM_PROMPT, user, legal_tools.TOOL_SPECS, dispatch)

    rec = tk.record
    address_confirmed = bool(tk.address)
    flags: list[dict[str, str]] = []
    if not address_confirmed:
        flags.append(_flag("address_unconfirmed", "high",
                           "Subject address not provided; legal identity could not be confirmed."))
    if rec.get("title_status") == "Encumbered":
        flags.append(_flag("title_encumbrance", "medium",
                           "Registered caveat/encumbrance on title; confirm it does not impair marketability."))
    for esc in tk.escalations:
        flags.append(_flag("legal_escalation", esc["severity"], esc["reason"]))

    documents = dict(state.get("documents", {}))
    legal_title = {
        "address_confirmed": address_confirmed,
        "legal_description": rec.get("legal_description", "Unverified") if address_confirmed else "Unverified",
        "parcel_id": rec.get("parcel_id", "Unverified") if address_confirmed else "Unverified",
        "title_status": rec.get("title_status", "Unverified") if address_confirmed else "Unverified",
        "registered_owner": rec.get("registered_owner", "On file with KV Capital"),
        "roll_number": rec.get("roll_number"),
        "encumbrances": rec.get("encumbrances", []),
        "conflicts": tk.conflicts,
        "flags": flags,
    }

    tools_used = [c["tool"] for c in run["calls"]]
    trace = state.get("trace", []) + [
        f"legal_title (LLM agent): address_confirmed={address_confirmed}, "
        f"parcel={legal_title['parcel_id']}, title={legal_title['title_status']}, "
        f"conflicts={len(tk.conflicts)}, flags={len(flags)}; tools={tools_used}"
    ]
    return {
        "legal_title": legal_title,
        "documents": documents,
        "evidence": state.get("evidence", []) + tk.evidence,
        "trace": trace,
    }
