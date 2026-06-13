"""Single source of truth for the synthetic Alberta market model, the agent's
adjustment coefficients, ranking weights, retrieval parameters, and risk
thresholds.

Both the data generator (`data/generate_data.py`) and the agent import from
here. Keeping the generator and the agent's coefficients aligned is what makes
the recovered valuation checkable: the agent is, in effect, inverting the same
pricing model used to synthesize sales.
"""

from __future__ import annotations

from datetime import date

# ---------------------------------------------------------------------------
# Market anchor / time
# ---------------------------------------------------------------------------
# Valuation "as of" date. Sales are synthesized over the 24 months prior.
VALUATION_DATE = date(2026, 6, 1)
SALES_HISTORY_MONTHS = 24

# Market appreciation used both to synthesize a time trend and to time-adjust
# comparable sales to the valuation date (~5%/yr).
MONTHLY_APPRECIATION = 0.004

# ---------------------------------------------------------------------------
# Geography: neighborhood centroids + base detached price-per-sqft ($/sqft)
# ---------------------------------------------------------------------------
# base_ppsf is the price-per-sqft for a baseline detached home in the
# neighborhood; property-type multipliers below scale it.
NEIGHBORHOODS: dict[str, dict] = {
    # Calgary
    "Mount Royal": {"city": "Calgary", "lat": 51.0290, "lon": -114.0890, "base_ppsf": 720},
    "Beltline": {"city": "Calgary", "lat": 51.0392, "lon": -114.0719, "base_ppsf": 560},
    "Kensington": {"city": "Calgary", "lat": 51.0535, "lon": -114.0900, "base_ppsf": 600},
    "Bridgeland": {"city": "Calgary", "lat": 51.0545, "lon": -114.0470, "base_ppsf": 560},
    "Inglewood": {"city": "Calgary", "lat": 51.0370, "lon": -114.0290, "base_ppsf": 540},
    "Signal Hill": {"city": "Calgary", "lat": 51.0700, "lon": -114.1700, "base_ppsf": 470},
    "Tuscany": {"city": "Calgary", "lat": 51.1230, "lon": -114.2400, "base_ppsf": 430},
    "Auburn Bay": {"city": "Calgary", "lat": 50.8830, "lon": -113.9560, "base_ppsf": 450},
    "Mahogany": {"city": "Calgary", "lat": 50.8930, "lon": -113.9290, "base_ppsf": 460},
    "Evanston": {"city": "Calgary", "lat": 51.1700, "lon": -114.0900, "base_ppsf": 410},
    # Edmonton
    "Glenora": {"city": "Edmonton", "lat": 53.5380, "lon": -113.5430, "base_ppsf": 520},
    "Oliver": {"city": "Edmonton", "lat": 53.5410, "lon": -113.5160, "base_ppsf": 430},
    "Garneau": {"city": "Edmonton", "lat": 53.5230, "lon": -113.5230, "base_ppsf": 440},
    "Old Strathcona": {"city": "Edmonton", "lat": 53.5180, "lon": -113.4960, "base_ppsf": 450},
    "Westmount": {"city": "Edmonton", "lat": 53.5560, "lon": -113.5320, "base_ppsf": 400},
    "Bonnie Doon": {"city": "Edmonton", "lat": 53.5230, "lon": -113.4640, "base_ppsf": 390},
    "Windermere": {"city": "Edmonton", "lat": 53.4350, "lon": -113.6300, "base_ppsf": 470},
    "Terwillegar": {"city": "Edmonton", "lat": 53.4540, "lon": -113.5840, "base_ppsf": 410},
    "Summerside": {"city": "Edmonton", "lat": 53.4180, "lon": -113.4350, "base_ppsf": 360},
    "Westmount Edmonton": {"city": "Edmonton", "lat": 53.5560, "lon": -113.5320, "base_ppsf": 400},
}
# Drop the accidental duplicate key kept above for readability.
NEIGHBORHOODS.pop("Westmount Edmonton", None)

# ---------------------------------------------------------------------------
# Property types
# ---------------------------------------------------------------------------
PROPERTY_TYPES = ["Detached", "Semi-Detached", "Townhouse", "Condo"]

# Multiplier applied to neighborhood base_ppsf by property type.
TYPE_PPSF_MULTIPLIER: dict[str, float] = {
    "Detached": 1.00,
    "Semi-Detached": 0.92,
    "Townhouse": 0.85,
    "Condo": 0.80,
}

# Whether a property type meaningfully carries private lot value.
TYPE_HAS_LOT: dict[str, bool] = {
    "Detached": True,
    "Semi-Detached": True,
    "Townhouse": False,
    "Condo": False,
}

# ---------------------------------------------------------------------------
# Additive adjustment coefficients (today's dollars)
# ---------------------------------------------------------------------------
# Marginal contributory value of features, used by the sales-comparison
# adjustment step. GLA is adjusted using neighborhood/type implied $/sqft (see
# valuation_math), so it is not a flat constant here.
BED_VALUE = 12_000          # $ per bedroom difference
BATH_VALUE = 9_000          # $ per bathroom difference
LOT_PPSF = 35               # $ per sqft of lot-size difference (lot-bearing types)
AGE_VALUE_PER_YEAR = 1_200  # $ per year of effective-age difference
MAX_AGE_FOR_DEPRECIATION = 60  # years; depreciation flattens beyond this

# ---------------------------------------------------------------------------
# Ranking weights (similarity). Higher weight = more influence on comp score.
# ---------------------------------------------------------------------------
RANKING_WEIGHTS: dict[str, float] = {
    "distance": 0.30,
    "recency": 0.20,
    "gla": 0.20,
    "beds_baths": 0.10,
    "same_type": 0.12,
    "same_neighborhood": 0.08,
}

# Normalizers controlling how quickly similarity decays with each difference.
DISTANCE_HALF_KM = 2.0       # distance at which the distance score ~0.5
RECENCY_HALF_MONTHS = 6.0    # sale age at which the recency score ~0.5
GLA_HALF_FRACTION = 0.15     # |GLA delta| fraction at which gla score ~0.5

# Number of comps selected for the valuation.
TOP_N_COMPS = 5

# ---------------------------------------------------------------------------
# Retrieval parameters (with progressive widening if too few candidates)
# ---------------------------------------------------------------------------
RETRIEVAL = {
    "radius_km": 3.0,
    "recency_months": 12,
    "gla_band": 0.30,           # +/- 30% of subject GLA
    "min_candidates": 8,
    "max_candidates": 60,
    # Multiplicative widening applied per attempt when below min_candidates.
    "widen_radius_factor": 1.8,
    "widen_recency_add_months": 6,
    "widen_gla_band_add": 0.15,
    "max_widen_attempts": 3,
}

# ---------------------------------------------------------------------------
# Data-quality: fields required to value at all vs. nice-to-have.
# ---------------------------------------------------------------------------
CRITICAL_FIELDS = ["property_type", "gla_sqft", "neighborhood"]
SOFT_FIELDS = ["bedrooms", "bathrooms", "lot_size_sqft", "year_built", "address"]

# ---------------------------------------------------------------------------
# Risk thresholds -> confidence
# ---------------------------------------------------------------------------
RISK = {
    "min_comps": 3,                 # fewer selected comps -> flag
    "cov_high": 0.12,               # coeff. of variation of adjusted values
    "cov_elevated": 0.07,
    "gross_adj_high": 0.30,         # mean gross adjustment as % of sale price
    "gross_adj_elevated": 0.18,     # appraisal practice: gross > ~25-30% is a concern
    "stale_months": 9,              # median comp age beyond this -> flag
    "far_km": 5.0,                  # median comp distance beyond this -> flag
    "data_quality_min": 0.6,        # below this quality score -> flag
}

# Confidence -> whether human review is required before reporting.
HUMAN_REVIEW_ON = ["Low"]  # confidence levels that force human review

# ---------------------------------------------------------------------------
# Case-folder layout
# ---------------------------------------------------------------------------
# A "case" is a single appraisal assignment delivered as a structured folder
# (see README / valuation_case_001). Each numbered section maps onto one or more
# pipeline nodes. Paths are relative to the case root. Structured files
# (csv/json/geojson/md) are parsed for real; the heavy formats (pdf/jpg/tif/
# docx/xlsx) are exposed as typed "document hooks" that the per-node tools fill
# in later.
CASE_LAYOUT: dict[str, dict] = {
    "assignment": {
        "dir": "00_assignment",
        "data": {"effective_date": "effective_date.json"},
        "documents": {
            "assignment_order": "assignment_order.pdf",
            "lender_instructions": "lender_instructions.pdf",
            "borrower_application": "borrower_application.pdf",
        },
    },
    "subject": {
        "dir": "01_subject_property",
        "data": {"listing": "subject_listing.csv"},
        "documents": {
            "subject_listing_pdf": "subject_listing.pdf",
            "purchase_contract": "purchase_contract.pdf",
            "floor_plan": "floor_plan.pdf",
            "rms_measurement_report": "rms_measurement_report.pdf",
            "inspection_notes": "inspection_notes.pdf",
        },
        "document_dirs": {"subject_photos": "subject_photos"},
    },
    # 02 legal/title + 03 assessment/tax are folded into the legal_title node.
    "legal_title": {
        "dir": "02_legal_title",
        "data": {},
        "documents": {
            "current_certificate_of_title": "current_certificate_of_title.pdf",
            "historical_title_search": "historical_title_search.pdf",
            "registered_plan": "registered_plan.tif",
            "real_property_report": "real_property_report.pdf",
            "compliance_certificate": "compliance_certificate.pdf",
        },
        "document_dirs": {"encumbrances": "encumbrances"},
    },
    "assessment_tax": {
        "dir": "03_assessment_tax",
        "data": {"assessment_open_data": "assessment_open_data.csv"},
        "documents": {
            "assessment_details_report": "assessment_details_report.pdf",
            "property_summary_report": "property_summary_report.pdf",
            "tax_certificate": "tax_certificate.pdf",
        },
    },
    # 04 zoning/land-use + 05 permits/rpr/surveys are folded into zoning_hbu.
    "zoning": {
        "dir": "04_zoning_land_use",
        "data": {
            "land_use_district": "land_use_district.geojson",
            "parcel_boundary": "parcel_boundary.geojson",
            "community_boundary": "community_boundary.geojson",
            "overlays": "overlays.geojson",
            "zoning_hbu_notes": "zoning_hbu_notes.md",
        },
        "documents": {"zoning_bylaw_excerpt": "zoning_bylaw_excerpt.pdf"},
    },
    "permits": {
        "dir": "05_permits_rpr_surveys",
        "data": {
            "building_permits": "building_permits.csv",
            "development_permits": "development_permits.csv",
            "survey_measurements": "survey_measurements.csv",
        },
        "documents": {},
        "document_dirs": {"permit_documents": "permit_documents"},
    },
    "comparables": {
        "dir": "06_comparables",
        "data": {
            "sold_comps_raw": "sold_comps_raw.csv",
            "active_listings_raw": "active_listings_raw.csv",
            "conditional_pending_raw": "conditional_pending_raw.csv",
            "rejected_comps": "rejected_comps.csv",
            "verified_comps": "verified_comps.csv",
        },
        "documents": {},
        "document_dirs": {
            "comp_photos": "comp_photos",
            "comp_listing_sheets": "comp_listing_sheets",
        },
    },
    "market": {
        "dir": "07_market_context",
        "data": {
            "price_index": "price_index.csv",
            "absorption_inventory": "absorption_inventory.csv",
            "days_on_market_stats": "days_on_market_stats.csv",
        },
        "documents": {
            "creb_monthly_market_report": "creb_monthly_market_report.pdf",
            "neighbourhood_market_summary": "neighbourhood_market_summary.pdf",
        },
    },
    "workflow_outputs": {"dir": "08_workflow_outputs", "data": {}, "documents": {}},
    "final_package": {"dir": "09_final_package", "data": {}, "documents": {}},
}

# Canonical internal comp/subject record schema. Case CSVs are normalized to
# these column names via COLUMN_ALIASES before the rest of the agent sees them.
CANONICAL_COMP_COLUMNS = [
    "id", "address", "city", "neighborhood", "lat", "lon", "property_type",
    "bedrooms", "bathrooms", "gla_sqft", "lot_size_sqft", "year_built",
    "sale_date", "sale_price",
]

# Maps a canonical column -> the set of header names that may appear in real
# case CSVs (matched case-insensitively, ignoring spaces/underscores). The
# canonical name itself is always accepted.
COLUMN_ALIASES: dict[str, list[str]] = {
    "id": ["id", "mls", "mls_number", "mls#", "listing_id", "comp_id"],
    "address": ["address", "street_address", "full_address", "property_address"],
    "city": ["city", "municipality"],
    "neighborhood": ["neighborhood", "neighbourhood", "community", "community_name", "area"],
    "lat": ["lat", "latitude", "y"],
    "lon": ["lon", "lng", "long", "longitude", "x"],
    "property_type": ["property_type", "type", "dwelling_type", "style", "prop_type"],
    "bedrooms": ["bedrooms", "beds", "bed", "br", "num_bedrooms"],
    "bathrooms": ["bathrooms", "baths", "bath", "ba", "num_bathrooms"],
    "gla_sqft": ["gla_sqft", "gla", "living_area", "sqft", "square_feet", "rms_area", "above_grade_sqft"],
    "lot_size_sqft": ["lot_size_sqft", "lot_size", "lot", "lot_sqft", "lot_area"],
    "year_built": ["year_built", "yr_built", "built", "vintage", "effective_year_built"],
    "sale_date": ["sale_date", "sold_date", "close_date", "closing_date", "date_sold", "list_date"],
    "sale_price": ["sale_price", "sold_price", "close_price", "price", "list_price", "sale_amount"],
}
