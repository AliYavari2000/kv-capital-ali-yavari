"""Node 2 - Subject Property Inspection / Measurement (LLM agent).

The agent reads the subject's listing facts, classifies property type / condition
/ build quality, detects renovations and site features, checks measurement
conflicts across sources, and runs the deterministic schema + data-quality gate
that decides whether the pipeline can proceed. Measurement math is deterministic
(see ``subject_tools.normalize_lot_size``).

No deterministic fallback: requires an LLM (see ``llm.run_tool_agent``).
"""

from __future__ import annotations

from typing import Any

from src import llm
from src import fast_path
from src.state import CompState
from src.tools import subject_tools

_SYSTEM_PROMPT = (
    "You are the Subject Property inspection agent for KV Capital's residential "
    "valuation workflow. Using ONLY the tools, understand the subject property: "
    "physical characteristics, condition, build quality, size/rooms, improvements, "
    "and site features. Do not invent measurements.\n\n"
    "Steps: call extract_subject_from_listing; inspect_photos and parse_floor_plan "
    "to see what visual evidence exists; classify_property_type, classify_condition, "
    "and classify_quality; detect_renovations and extract_site_features from the "
    "evidence; normalize_measurement for any area not already in square feet; "
    "check_measurement_conflicts; then ALWAYS call validate_subject_schema. "
    "append_evidence for each source. If size, condition, or property type is "
    "unclear or conflicting, call raise_human_review. Cite Calgary/Alberta "
    "authorities (Pillar 9/MLS listing, RECA RMS report, Open Calgary assessment) "
    "via source_id in append_evidence. Finish with a one-sentence summary of the subject."
)


def assignment_user_prompt(tk: subject_tools.SubjectToolkit) -> str:
    import json
    return (
        "Inspect this subject. Facts already on file:\n"
        + json.dumps(tk.extract_subject_from_listing(), indent=2)
        + "\n\nUse the tools to classify, normalize, and validate it."
    )


def subject_property_node(state: CompState) -> dict[str, Any]:
    if not llm.llm_available():
        raise llm.LLMUnavailableError(
            "Subject Property is an LLM agent and requires OPENAI_API_KEY to be set."
        )

    tk = subject_tools.SubjectToolkit(state)
    dispatch = subject_tools.build_dispatch(tk)
    run, mode = llm.run_node_agent(
        state, dispatch, subject_tools.TOOL_SPECS, _SYSTEM_PROMPT, assignment_user_prompt(tk),
        lambda: fast_path.run_subject(tk, dispatch),
    )

    # Guarantee the deterministic gate is computed even if the agent skipped it.
    if not tk.data_quality:
        tk.validate_subject_schema()

    dq = dict(tk.data_quality)
    if tk.conflicts and dq.get("passed"):
        dq["issues"] = list(dq.get("issues", [])) + [
            f"measurement conflict on {c['field']}" for c in tk.conflicts]

    tools_used = [c["tool"] for c in run["calls"]]
    trace = state.get("trace", []) + [
        f"subject_property ({mode}): type={tk.subject.get('property_type')}, "
        f"condition={tk.subject.get('condition')}, quality={tk.subject.get('quality')}, "
        f"conflicts={len(tk.conflicts)}; dq score={dq.get('score')}, passed={dq.get('passed')}, "
        f"missing_critical={dq.get('missing_critical') or 'none'}; tools={tools_used}"
    ]
    return {
        "subject": tk.subject,
        "data_quality": dq,
        "evidence": state.get("evidence", []) + tk.evidence,
        "trace": trace,
    }
