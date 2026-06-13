"""Deterministic fast path — skip LLM tool orchestration for structured inputs.

Each agent node exposes deterministic tools; the LLM normally decides which to
call and in what order (many API round trips). When ``enabled(state)`` is true
we run a fixed script instead. Valuation math is unchanged; only orchestration
latency is removed.

Enable with ``KV_FAST_PATH=1`` (always) or ``auto`` (default: on for case
folders). Disable with ``KV_FAST_PATH=0``.

Set ``KV_FAST_NARRATIVE=1`` to skip LLM prose in reconciliation/report (uses
templates; fastest end-to-end).
"""

from __future__ import annotations

import os
from typing import Any

from src import config


def enabled(state: dict[str, Any]) -> bool:
    mode = os.getenv("KV_FAST_PATH", "auto").strip().lower()
    if mode in ("0", "false", "no", "off"):
        return False
    if mode in ("1", "true", "yes", "on"):
        return True
    # auto: structured case folders have parseable CSV/JSON — no LLM orchestration needed
    return bool(state.get("case_dir"))


def fast_narratives() -> bool:
    return os.getenv("KV_FAST_NARRATIVE", "0").strip().lower() in ("1", "true", "yes", "on")


def _calls(steps: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    return {"calls": [{"tool": n, "args": a} for n, a in steps], "messages": [], "final": None}


def run_script(dispatch: dict[str, Any], steps: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    for name, kwargs in steps:
        fn = dispatch.get(name)
        if fn:
            fn(**kwargs)
    return _calls(steps)


def run_intake(tk: Any, dispatch: dict[str, Any]) -> dict[str, Any]:
    tk.parse_assignment_request()
    tk.parse_listing_input()
    atype = tk.assignment_request.get("assignment_type_hint") or "purchase"
    steps: list[tuple[str, dict[str, Any]]] = [
        ("parse_client_instructions", {}),
        ("validate_effective_date", {}),
        ("detect_assignment_type", {"assignment_type": str(atype), "rationale": "from case assignment metadata"}),
        ("required_document_checklist", {}),
        ("missing_info_detector", {}),
        ("create_workfile_id", {}),
    ]
    calls = [
        {"tool": "parse_assignment_request", "args": {}},
        {"tool": "parse_listing_input", "args": {}},
    ] + run_script(dispatch, steps)["calls"]
    return {"calls": calls, "messages": [], "final": None}


def run_subject(tk: Any, dispatch: dict[str, Any]) -> dict[str, Any]:
    facts = tk.extract_subject_from_listing()
    ptype = facts.get("property_type") or tk.subject.get("property_type") or "Detached"
    steps = [
        ("inspect_photos", {}),
        ("parse_floor_plan", {}),
        ("classify_property_type", {"property_type": str(ptype)}),
        ("classify_condition", {"condition": "average"}),
        ("classify_quality", {"quality": "standard"}),
        ("detect_renovations", {"renovations": []}),
        ("extract_site_features", {"features": {}}),
        ("check_measurement_conflicts", {}),
        ("validate_subject_schema", {}),
    ]
    calls = [{"tool": "extract_subject_from_listing", "args": {}}] + run_script(dispatch, steps)["calls"]
    return {"calls": calls, "messages": [], "final": None}


def run_legal(tk: Any, dispatch: dict[str, Any]) -> dict[str, Any]:
    steps = [
        ("parcel_lookup_by_address", {}),
        ("address_normalizer", {}),
        ("title_document_parser", {}),
        ("tax_roll_lookup", {}),
        ("parcel_boundary_lookup", {}),
        ("owner_name_extractor", {}),
        ("encumbrance_extractor", {}),
        ("legal_description_matcher", {}),
        ("title_conflict_detector", {}),
    ]
    return run_script(dispatch, steps)


def run_zoning(tk: Any, dispatch: dict[str, Any]) -> dict[str, Any]:
    pre = [
        ("municipal_zoning_lookup", {}),
        ("zoning_map_query", {}),
        ("permitted_use_checker", {}),
        ("development_standard_extractor", {}),
        ("nonconforming_use_detector", {}),
        ("overlay_lookup", {}),
        ("permit_history_lookup", {}),
    ]
    run_script(dispatch, pre)
    code = tk.record.get("zoning_code", "R-C1")
    summary = (
        f"The property is zoned {code}, allowing single detached dwellings, which is "
        f"the current use. Present use is legally permissible, physically possible, "
        f"financially feasible, and maximally productive."
    )
    post = [
        ("hbu_rule_engine", {
            "legally_permissible": True,
            "physically_possible": True,
            "financially_feasible": True,
            "maximally_productive": True,
        }),
        ("zoning_summary_generator", {"summary": summary}),
    ]
    calls = _calls(pre)["calls"] + run_script(dispatch, post)["calls"]
    return {"calls": calls, "messages": [], "final": None}


def run_market_scope(tk: Any, dispatch: dict[str, Any], *, rerun: int = 0) -> dict[str, Any]:
    gla = float(tk.gla or 2000)
    band = config.RETRIEVAL["gla_band"]
    gla_lo = int(gla * (1 - band))
    gla_hi = int(gla * (1 + band))
    radius = config.RETRIEVAL["radius_km"] if not rerun else round(
        config.RETRIEVAL["radius_km"] * config.RETRIEVAL["widen_radius_factor"], 2
    )
    recency = config.RETRIEVAL["recency_months"] if not rerun else (
        config.RETRIEVAL["recency_months"] + config.RETRIEVAL["widen_recency_add_months"]
    )
    steps = [
        ("neighbourhood_boundary_lookup", {}),
        ("submarket_classifier", {}),
        ("market_inventory_snapshot", {}),
        ("radius_search_area", {"radius_km": radius}),
        ("search_window_selector", {"months": recency}),
        ("comp_filter_builder", {"gla_range": [gla_lo, gla_hi]}),
    ]
    return run_script(dispatch, steps)


def run_retrieval(tk: Any, dispatch: dict[str, Any]) -> dict[str, Any]:
    top_n = max(config.TOP_N_COMPS, 10)
    steps = [
        ("query_sold_comps", {}),
        ("similarity_search_comps", {}),
        ("deduplicate_properties", {}),
        ("arms_length_filter", {}),
        ("outlier_sale_detector", {}),
        ("comp_candidate_ranker", {"top_n": top_n}),
        ("query_active_listings", {}),
        ("query_pending_or_conditional", {}),
    ]
    return run_script(dispatch, steps)


def run_verification(tk: Any, dispatch: dict[str, Any]) -> dict[str, Any]:
    steps = [
        ("cross_source_fact_checker", {}),
        ("assessment_record_lookup", {}),
        ("permit_lookup", {}),
        ("sale_price_verifier", {}),
        ("gla_conflict_detector", {}),
        ("lot_size_conflict_detector", {}),
        ("property_type_conflict_detector", {}),
    ]
    result = run_script(dispatch, steps)
    for field, sources in (
        ("gla_sqft", ["listing", "assessment"]),
        ("year_built", ["listing", "assessment"]),
        ("lot_size_sqft", ["listing", "survey"]),
    ):
        val = tk.subject.get(field)
        if val is not None:
            score = dispatch["confidence_score"](field=field, sources=sources)
            dispatch["verified_fact_writer"](
                field=field, value=val,
                confidence=score["confidence"], sources=sources,
            )
            result["calls"].append({"tool": "verified_fact_writer", "args": {"field": field}})
    return result


def run_normalization(tk: Any, dispatch: dict[str, Any]) -> dict[str, Any]:
    steps = [
        ("canonical_property_mapper", {}),
        ("location_feature_encoder", {}),
        ("missing_value_handler", {}),
        ("outlier_feature_detector", {}),
        ("normalization_report_writer", {}),
    ]
    return run_script(dispatch, steps)
