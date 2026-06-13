"""Calgary / Alberta data-source registry for the residential valuation workflow.

Each authoritative source used in a real Alberta appraisal is catalogued here and
mapped onto the case-folder layout (``config.CASE_LAYOUT``). When a case is
loaded, ``case_store`` annotates every structured file and document hook with
the matching source metadata so agents can cite provenance in evidence and the
report writer can list what was relied on.

This fixture uses synthetic/placeholder files that *stand in for* exports from
these real systems (Open Calgary, SPIN2, Pillar 9, CREB, etc.).
"""

from __future__ import annotations

from typing import Any, Optional

# ---------------------------------------------------------------------------
# Authoritative sources (Calgary / Alberta residential appraisal)
# ---------------------------------------------------------------------------
DATA_SOURCES: dict[str, dict[str, Any]] = {
    "cuspap": {
        "name": "CUSPAP (Canadian Uniform Standards of Professional Appraisal Practice)",
        "provider": "Appraisal Institute of Canada (AIC)",
        "format": "PDF / web",
        "jurisdiction": "Canada",
        "why_it_matters": (
            "Governs appraisal practice requirements; AIC members must comply with CUSPAP."
        ),
        "url": "https://www.aicanada.ca/",
    },
    "reca_rms": {
        "name": "RECA Residential Measurement Standard (RMS)",
        "provider": "Real Estate Council of Alberta (RECA)",
        "format": "PDF / web",
        "jurisdiction": "Alberta",
        "why_it_matters": (
            "Alberta licensees must use RMS when measuring/advertising residential "
            "property; detached homes are measured using the exterior wall at foundation."
        ),
        "url": "https://www.reca.ca/",
    },
    "calgary_assessment_pdf": {
        "name": "Calgary Assessment Details Report",
        "provider": "City of Calgary Assessment",
        "format": "PDF",
        "jurisdiction": "Calgary",
        "why_it_matters": (
            "Contains current assessment, property type, property details, use, class, "
            "site influences, title number, legal description, LINC, declared value, "
            "land use, and permit existence."
        ),
        "url": "https://assessment.calgary.ca/",
    },
    "open_calgary_assessment": {
        "name": "Open Calgary — Current & Historical Assessment",
        "provider": "City of Calgary (Open Data)",
        "format": "CSV / API",
        "jurisdiction": "Calgary",
        "why_it_matters": "Calgary publishes current and historical parcel assessment datasets.",
        "url": "https://data.calgary.ca/",
    },
    "open_calgary_parcel": {
        "name": "Open Calgary — Parcel Address / OPF",
        "provider": "City of Calgary (Open Data)",
        "format": "CSV / GeoJSON",
        "jurisdiction": "Calgary",
        "why_it_matters": (
            "Needed for address normalization, parcel geometry, plan/block/lot, and area."
        ),
        "url": "https://data.calgary.ca/",
    },
    "open_calgary_zoning": {
        "name": "Open Calgary — Land Use Districts",
        "provider": "City of Calgary (Open Data)",
        "format": "GeoJSON / SHP / CSV",
        "jurisdiction": "Calgary",
        "why_it_matters": "Needed to check legal use, zoning district, and highest-and-best-use.",
        "url": "https://data.calgary.ca/",
    },
    "open_calgary_community": {
        "name": "Open Calgary — Community Boundaries",
        "provider": "City of Calgary (Open Data)",
        "format": "GeoJSON / SHP / CSV",
        "jurisdiction": "Calgary",
        "why_it_matters": "Needed to define neighbourhood/submarket and comp search boundaries.",
        "url": "https://data.calgary.ca/",
    },
    "open_calgary_building_permits": {
        "name": "Open Calgary — Building Permits",
        "provider": "City of Calgary (Open Data)",
        "format": "CSV / API",
        "jurisdiction": "Calgary",
        "why_it_matters": (
            "Needed to verify renovations, basement development, additions, garages, etc."
        ),
        "url": "https://data.calgary.ca/",
    },
    "open_calgary_development_permits": {
        "name": "Open Calgary — Development Permits",
        "provider": "City of Calgary (Open Data)",
        "format": "CSV / API",
        "jurisdiction": "Calgary",
        "why_it_matters": (
            "Needed for legality, land-use changes, major redevelopments, discretionary approvals."
        ),
        "url": "https://data.calgary.ca/",
    },
    "calgary_zoning_bylaw": {
        "name": "Calgary Land Use Bylaw",
        "provider": "City of Calgary",
        "format": "PDF / web",
        "jurisdiction": "Calgary",
        "why_it_matters": "Authoritative text for permitted uses, setbacks, and development standards.",
        "url": "https://www.calgary.ca/planning/land-use/bylaws.html",
    },
    "alberta_spin2_title": {
        "name": "Alberta SPIN2 — Certificate of Title",
        "provider": "Alberta Land Titles (SPIN2)",
        "format": "PDF",
        "jurisdiction": "Alberta",
        "why_it_matters": (
            "Current title is available as a certified PDF and describes ownership "
            "rights and registered instruments/caveats."
        ),
        "url": "https://www.alberta.ca/spin2",
    },
    "alberta_spin2_survey": {
        "name": "Alberta SPIN2 — Registered Survey Plan",
        "provider": "Alberta Land Titles (SPIN2)",
        "format": "TIF / paper / digital",
        "jurisdiction": "Alberta",
        "why_it_matters": (
            "Registered survey plans are available through SPIN2 or registry agents; "
            "digital copies are TIF."
        ),
        "url": "https://www.alberta.ca/spin2",
    },
    "rpr_survey": {
        "name": "Real Property Report (RPR)",
        "provider": "Alberta Land Surveyor",
        "format": "PDF",
        "jurisdiction": "Alberta",
        "why_it_matters": "Shows improvements relative to parcel boundaries for compliance and area checks.",
    },
    "pillar9_mls": {
        "name": "Pillar 9 / MLS / REALTOR Systems",
        "provider": "Alberta MLS (Pillar 9)",
        "format": "CSV / PDF / images",
        "jurisdiction": "Alberta",
        "why_it_matters": (
            "Needed for real comparable sales, listing remarks, DOM, photos, sale "
            "conditions, and listing history. Pillar 9 aggregates Alberta listing data."
        ),
        "url": "https://www.pillar9.ca/",
    },
    "creb_market": {
        "name": "CREB Monthly Market Report",
        "provider": "Calgary Real Estate Board (CREB)",
        "format": "PDF / CSV",
        "jurisdiction": "Calgary region",
        "why_it_matters": (
            "Needed for time adjustments, market direction, inventory, benchmark "
            "price, sales/new listings ratio."
        ),
        "url": "https://www.creb.com/",
    },
    "kv_internal": {
        "name": "KV Capital Assignment Package",
        "provider": "KV Capital Credit",
        "format": "PDF / internal",
        "jurisdiction": "Alberta",
        "why_it_matters": "Lender assignment order, borrower application, and engagement instructions.",
    },
}

# Maps (case section, file key) -> source id from DATA_SOURCES.
CASE_FILE_SOURCES: dict[tuple[str, str], str] = {
    # 00 assignment
    ("assignment", "assignment_order"): "kv_internal",
    ("assignment", "lender_instructions"): "kv_internal",
    ("assignment", "borrower_application"): "kv_internal",
    ("assignment", "effective_date"): "kv_internal",
    # 01 subject
    ("subject", "listing"): "pillar9_mls",
    ("subject", "subject_listing_pdf"): "pillar9_mls",
    ("subject", "purchase_contract"): "kv_internal",
    ("subject", "floor_plan"): "pillar9_mls",
    ("subject", "rms_measurement_report"): "reca_rms",
    ("subject", "inspection_notes"): "kv_internal",
    ("subject", "subject_photos"): "pillar9_mls",
    # 02 legal / title
    ("legal_title", "current_certificate_of_title"): "alberta_spin2_title",
    ("legal_title", "historical_title_search"): "alberta_spin2_title",
    ("legal_title", "registered_plan"): "alberta_spin2_survey",
    ("legal_title", "real_property_report"): "rpr_survey",
    ("legal_title", "compliance_certificate"): "rpr_survey",
    ("legal_title", "encumbrances"): "alberta_spin2_title",
    # 03 assessment / tax
    ("assessment_tax", "assessment_open_data"): "open_calgary_assessment",
    ("assessment_tax", "assessment_details_report"): "calgary_assessment_pdf",
    ("assessment_tax", "property_summary_report"): "calgary_assessment_pdf",
    ("assessment_tax", "tax_certificate"): "calgary_assessment_pdf",
    # 04 zoning
    ("zoning", "land_use_district"): "open_calgary_zoning",
    ("zoning", "parcel_boundary"): "open_calgary_parcel",
    ("zoning", "community_boundary"): "open_calgary_community",
    ("zoning", "overlays"): "open_calgary_zoning",
    ("zoning", "zoning_hbu_notes"): "open_calgary_zoning",
    ("zoning", "zoning_bylaw_excerpt"): "calgary_zoning_bylaw",
    # 05 permits
    ("permits", "building_permits"): "open_calgary_building_permits",
    ("permits", "development_permits"): "open_calgary_development_permits",
    ("permits", "survey_measurements"): "rpr_survey",
    ("permits", "permit_documents"): "open_calgary_building_permits",
    # 06 comparables
    ("comparables", "sold_comps_raw"): "pillar9_mls",
    ("comparables", "active_listings_raw"): "pillar9_mls",
    ("comparables", "conditional_pending_raw"): "pillar9_mls",
    ("comparables", "rejected_comps"): "pillar9_mls",
    ("comparables", "verified_comps"): "pillar9_mls",
    ("comparables", "comp_photos"): "pillar9_mls",
    ("comparables", "comp_listing_sheets"): "pillar9_mls",
    # 07 market
    ("market", "price_index"): "creb_market",
    ("market", "absorption_inventory"): "creb_market",
    ("market", "days_on_market_stats"): "creb_market",
    ("market", "creb_monthly_market_report"): "creb_market",
    ("market", "neighbourhood_market_summary"): "creb_market",
}

# Standards referenced by practice (not always a case file).
PRACTICE_STANDARDS = ("cuspap", "reca_rms")

JURISDICTION = "Calgary, Alberta, Canada"


def get_source(source_id: str) -> dict[str, Any]:
    """Return a source record, or a minimal stub if unknown."""
    rec = DATA_SOURCES.get(source_id)
    if rec:
        return {"id": source_id, **rec}
    return {"id": source_id, "name": source_id, "provider": "unknown", "format": "unknown",
            "why_it_matters": ""}


def source_for_case_file(section: str, file_key: str) -> dict[str, Any]:
    """Look up the authoritative source for a case-folder file key."""
    sid = CASE_FILE_SOURCES.get((section, file_key))
    return get_source(sid) if sid else {"id": None, "name": "unmapped", "provider": "unknown"}


def annotate_hook(hook: dict[str, Any], section: str, file_key: str) -> dict[str, Any]:
    """Attach source metadata to a document hook."""
    src = source_for_case_file(section, file_key)
    out = dict(hook)
    out["source_id"] = src.get("id")
    out["source"] = src.get("name")
    out["provider"] = src.get("provider")
    out["source_format"] = src.get("format")
    out["why_it_matters"] = src.get("why_it_matters", "")
    return out


def annotate_data(data: Any, section: str, file_key: str) -> Any:
    """Wrap structured data with a lightweight provenance envelope."""
    src = source_for_case_file(section, file_key)
    if src.get("id") is None:
        return data
    return {
        "_provenance": {
            "source_id": src["id"],
            "source": src["name"],
            "provider": src["provider"],
            "format": src.get("format"),
            "why_it_matters": src.get("why_it_matters", ""),
        },
        "records": data,
    }


def unwrap_data(data: Any) -> Any:
    """Return bare records from a provenance envelope (or pass through)."""
    if isinstance(data, dict) and "_provenance" in data and "records" in data:
        return data["records"]
    return data


def format_citation(source_id: str, detail: str = "") -> str:
    """One-line citation string for evidence logs."""
    src = get_source(source_id)
    base = f"{src['name']} ({src['provider']})"
    return f"{base}: {detail}" if detail else base


def case_manifest() -> dict[str, Any]:
    """Full manifest suitable for writing into a case folder or graph state."""
    files = []
    for (section, key), sid in sorted(CASE_FILE_SOURCES.items()):
        src = get_source(sid)
        files.append({
            "section": section,
            "file_key": key,
            "source_id": sid,
            "source": src["name"],
            "provider": src["provider"],
            "format": src.get("format"),
            "why_it_matters": src.get("why_it_matters", ""),
        })
    return {
        "jurisdiction": JURISDICTION,
        "practice_standards": [get_source(s) for s in PRACTICE_STANDARDS],
        "sources": [get_source(sid) for sid in DATA_SOURCES],
        "case_file_mapping": files,
    }


def sources_for_node(node_name: str) -> list[dict[str, Any]]:
    """Return the authoritative sources a pipeline node is expected to rely on."""
    _NODE_SOURCES = {
        "assignment_intake": ["kv_internal", "pillar9_mls", "cuspap"],
        "subject_property": ["pillar9_mls", "reca_rms", "calgary_assessment_pdf"],
        "legal_title": ["alberta_spin2_title", "alberta_spin2_survey", "open_calgary_assessment",
                        "open_calgary_parcel", "calgary_assessment_pdf"],
        "zoning_hbu": ["open_calgary_zoning", "open_calgary_community", "calgary_zoning_bylaw",
                       "open_calgary_building_permits", "open_calgary_development_permits"],
        "market_scope": ["open_calgary_community", "creb_market", "pillar9_mls"],
        "comp_retrieval": ["pillar9_mls", "open_calgary_assessment"],
        "fact_verification": ["pillar9_mls", "open_calgary_assessment", "open_calgary_building_permits",
                              "rpr_survey", "alberta_spin2_title", "reca_rms"],
        "normalization": ["reca_rms"],
        "adjustment_engine": ["creb_market", "pillar9_mls"],
        "reconciliation": ["cuspap", "pillar9_mls"],
        "report_writer": list(DATA_SOURCES.keys()),
    }
    ids = _NODE_SOURCES.get(node_name, [])
    return [get_source(s) for s in ids]
