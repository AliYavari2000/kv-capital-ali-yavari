"""Node 1 - Assignment Intake (``AssignmentIntakeNode``) -- LLM agent.

This node is an LLM tool-calling agent. The model is given the intake toolkit
(see ``src/tools/intake_tools.py``) and decides how to use it to:

  * define the assignment: type, effective date, intended use, client/borrower,
    and any special lender requirements;
  * build the subject's preliminary fact set from the listing input;
  * determine the required-document checklist and flag missing information;
  * mint a workfile ID and log evidence;
  * escalate ambiguous assignments to a human.

There is no deterministic fallback: if ``OPENAI_API_KEY`` is not set the node
raises (see ``llm.run_tool_agent``).
"""

from __future__ import annotations

import json
from typing import Any

from src import config, llm
from src import data_sources as ds
from src import fast_path
from src.state import CompState
from src.tools import intake_tools

_SYSTEM_PROMPT = (
    "You are the Assignment Intake agent for KV Capital's residential valuation "
    "workflow. Your job is to define the appraisal assignment and prepare the "
    "subject's preliminary facts using ONLY the tools provided. Do not invent "
    "facts; read them with the tools.\n\n"
    "Work through these steps, calling tools as needed:\n"
    "1. parse_assignment_request and parse_listing_input to gather the assignment "
    "and property facts.\n"
    "2. parse_client_instructions for any special lender/client requirements.\n"
    "3. validate_effective_date on the effective date you found.\n"
    "4. detect_assignment_type (purchase, refinance, construction_loan, "
    "retrospective, estate, other) based on the evidence.\n"
    "5. required_document_checklist for that assignment type.\n"
    "6. missing_info_detector to flag gaps.\n"
    "7. create_workfile_id.\n"
    "8. append_evidence for each source you relied on (use source_id from "
    "list_data_sources when citing Calgary/Alberta authorities such as Open Calgary, "
    "SPIN2, Pillar 9/MLS, CREB, RECA RMS, or CUSPAP).\n"
    "9. If the assignment is ambiguous or critical info is missing, call "
    "raise_human_review with a clear reason.\n\n"
    "When finished, reply with a one-sentence summary of the assignment. "
    "Do not fabricate addresses, dates, or document contents."
)


def _build_user_prompt(toolkit: intake_tools.IntakeToolkit) -> str:
    """Describe what inputs are available so the agent knows where to look."""
    if toolkit.case_dir:
        avail = {
            "input_kind": "case_folder",
            "case_dir": toolkit.case_dir,
            "assignment_documents": sorted(toolkit._assignment_docs.keys()),
            "subject_documents": sorted(toolkit._subject_docs.keys()),
            "subject_listing_rows": len(toolkit._listing_rows),
        }
    elif isinstance(toolkit.raw_input, dict):
        avail = {"input_kind": "structured_dict", "keys": sorted(toolkit.raw_input.keys())}
    elif isinstance(toolkit.raw_input, str):
        avail = {"input_kind": "free_text_listing", "text_preview": toolkit.raw_input[:400]}
    else:
        avail = {"input_kind": "none"}
    return (
        "Define this valuation assignment. Available inputs:\n"
        + json.dumps(avail, indent=2)
        + "\n\nUse the tools to read these inputs and complete the intake."
    )


def assignment_intake_node(state: CompState) -> dict[str, Any]:
    # No fallback: the agent requires an LLM.
    if not llm.llm_available():
        raise llm.LLMUnavailableError(
            "Assignment Intake is an LLM agent and requires OPENAI_API_KEY to be set."
        )

    toolkit = intake_tools.IntakeToolkit(state)
    dispatch = intake_tools.build_dispatch(toolkit)

    run, mode = llm.run_node_agent(
        state, dispatch, intake_tools.TOOL_SPECS, _SYSTEM_PROMPT, _build_user_prompt(toolkit),
        lambda: fast_path.run_intake(toolkit, dispatch),
    )

    # Assemble graph state from what the agent's tool calls produced.
    ar = toolkit.assignment_request
    subject = dict(toolkit.listing)
    if not subject.get("address") and ar.get("subject_address"):
        subject["address"] = ar["subject_address"]

    effective_date = (
        toolkit.effective_date_check.get("normalized")
        or ar.get("effective_date")
        or config.VALUATION_DATE.isoformat()
    )
    report_date = ar.get("report_date") or config.VALUATION_DATE.isoformat()

    assignment = {
        "client": ar.get("client") or "KV Capital Credit",
        "borrower": ar.get("borrower") or "n/a",
        "intended_use": ar.get("valuation_purpose")
        or "Mortgage financing / collateral valuation",
        "valuation_purpose": ar.get("valuation_purpose"),
        "effective_date": str(effective_date),
        "report_date": str(report_date),
        "property_type": subject.get("property_type"),
        "assignment_type": toolkit.assignment_type.get("assignment_type"),
        "special_requirements": toolkit.instructions.get("special_requirements", []),
        "required_documents": toolkit.checklist,
        "missing_info": toolkit.missing_info.get("missing", []),
        "workfile_id": toolkit.workfile_id,
        "requires_human_review": bool(toolkit.escalations),
        "escalations": toolkit.escalations,
    }

    tools_used = [c["tool"] for c in run["calls"]]
    trace = state.get("trace", []) + [
        f"assignment_intake ({mode}): type={assignment['assignment_type']}, "
        f"effective_date={assignment['effective_date']}, workfile={toolkit.workfile_id}, "
        f"missing={assignment['missing_info'] or 'none'}, "
        f"escalations={len(toolkit.escalations)}; tools={tools_used}"
    ]

    out: dict[str, Any] = {
        "subject": subject,
        "assignment": assignment,
        "intake_source": "case_folder" if toolkit.case_dir else (
            "structured" if isinstance(toolkit.raw_input, dict) else "free_text"),
        "documents": toolkit.documents,
        "data_sources": toolkit.documents.get("data_source_manifest", ds.case_manifest()),
        "evidence": state.get("evidence", []) + toolkit.evidence,
        "trace": trace,
    }
    if toolkit.workfile_id:
        out["workfile_id"] = toolkit.workfile_id
    return out
