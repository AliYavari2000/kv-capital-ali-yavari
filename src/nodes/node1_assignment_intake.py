"""Node 1 - Assignment Intake (``AssignmentIntakeNode``).

Defines the assignment and effective date. Parses borrower/listing info into a
preliminary ``subject`` dict (free text via the LLM, with a deterministic regex
fallback) and records the assignment scope a real appraisal needs: client,
borrower, intended use, and the valuation ("effective") date.

Tools: form parser, LLM extractor.
"""

from __future__ import annotations

from typing import Any

from src import config, llm
from src.state import CompState



_FIELDS = [
    "address", "city", "neighborhood", "property_type",
    "bedrooms", "bathrooms", "gla_sqft", "lot_size_sqft", "year_built",
]




def assignment_intake_node(state: CompState) -> dict[str, Any]:
    raw = state.get("raw_input")
    source = "structured"
    if isinstance(raw, str):
        source = "free_text"
        parsed = llm.parse_listing(raw)
    elif isinstance(raw, dict):
        parsed = dict(raw)
    else:
        parsed = {}

    subject = {k: parsed.get(k) for k in _FIELDS if parsed.get(k) not in (None, "")}
    provided = sorted(subject.keys())

    # Assignment metadata: pull any overrides supplied with structured input,
    # otherwise fall back to sensible underwriting defaults.
    meta = parsed if isinstance(parsed, dict) else {}
    assignment = {
        "client": meta.get("client", "KV Capital Credit"),
        "borrower": meta.get("borrower", "n/a"),
        "intended_use": meta.get("intended_use", "Mortgage financing / collateral valuation"),
        "effective_date": str(meta.get("effective_date", config.VALUATION_DATE.isoformat())),
        "report_date": str(meta.get("report_date", config.VALUATION_DATE.isoformat())),
        "property_type": subject.get("property_type"),
    }

    trace = state.get("trace", []) + [
        f"assignment_intake: source={source}, effective_date={assignment['effective_date']}, "
        f"intended_use='{assignment['intended_use']}', parsed_fields={provided}"
        + ("" if llm.llm_available() or source != "free_text" else " (regex fallback)")
    ]
    return {
        "subject": subject,
        "assignment": assignment,
        "intake_source": source,
        "trace": trace,
    }
