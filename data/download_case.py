"""Download a real Calgary valuation case from Open Calgary public data.

Pulls authoritative City of Calgary open-data records (property assessment,
building permits, parcel geometry) and assembles them into the standard case
folder layout (``config.CASE_LAYOUT``). Comparable *sale* data is not publicly
available (MLS/Pillar 9 is licensed), so sold/active comp CSVs are filled from
the project's synthetic ``data/comps.csv`` filtered to the subject neighbourhood.

    python data/download_case.py                            # -> valuation_case_001/
    python data/download_case.py --community "TUSCANY" --address ""  # auto-pick subject
    python data/download_case.py --address "2952 SIGNAL HILL DR SW"
    python data/download_case.py --out valuation_case_custom

Sources (downloaded live):
  - Open Calgary Current Year Property Assessments (4bsw-nn7w)
  - Open Calgary Building Permits (c2es-76ed)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src import data_sources as ds  # noqa: E402
from src.valuation_math import haversine_km  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMPS = os.path.join(_ROOT, "data", "comps.csv")

_ASSESSMENT_API = "https://data.calgary.ca/resource/4bsw-nn7w.json"
_PERMITS_API = "https://data.calgary.ca/resource/c2es-76ed.json"

# Default second case: Signal Hill detached (different submarket from Tuscany).
DEFAULT_COMMUNITY = "SIGNAL HILL"
DEFAULT_ADDRESS = "2952 SIGNAL HILL DR SW"

# Open Calgary sub_property_use R110 = single detached dwelling.
_DETACHED_USE = "R110"


def _socrata_get(base: str, params: dict[str, str]) -> list[dict[str, Any]]:
    qs = urllib.parse.urlencode(params)
    url = f"{base}?{qs}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _touch(path: str, text: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _placeholder(path: str, kind: str, *, note: str = "") -> None:
    body = f"PLACEHOLDER {kind} for {os.path.basename(path)}.\n"
    if note:
        body += f"{note}\n"
    _touch(path, body)


def _write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Force identifier columns to strings so pandas/msgpack never see oversized ints.
    id_cols = {"roll_number", "roll_year", "unique_key", "cpid", "permit_number"}
    cleaned = []
    for row in rows:
        rec = dict(row)
        for col in id_cols:
            if col in rec and rec[col] is not None:
                rec[col] = str(rec[col])
        cleaned.append(rec)
    pd.DataFrame(cleaned).to_csv(path, index=False)


def _centroid(multipolygon: dict) -> tuple[float, float]:
    """Return (lat, lon) from the first ring of a MultiPolygon."""
    try:
        ring = multipolygon["coordinates"][0][0]
        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        return sum(lats) / len(lats), sum(lons) / len(lons)
    except (KeyError, IndexError, TypeError):
        return 0.0, 0.0


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> Optional[int]:
    f = _to_float(v)
    return int(round(f)) if f is not None else None


def _title_case_community(name: str) -> str:
    return " ".join(w.capitalize() for w in name.strip().split())


def _fetch_community_parcels(community: str, *, limit: int = 200) -> list[dict[str, Any]]:
    rows = _socrata_get(_ASSESSMENT_API, {
        "comm_name": community.upper(),
        "sub_property_use": _DETACHED_USE,
        "$limit": str(limit),
    })
    return [r for r in rows if r.get("sub_property_use") == _DETACHED_USE]


def _pick_subject(parcels: list[dict], address: Optional[str]) -> dict[str, Any]:
    if address:
        target = address.upper().strip()
        for row in parcels:
            if str(row.get("address", "")).upper().strip() == target:
                return row
        raise ValueError(f"Address not found in {len(parcels)} parcels: {address}")
    # Prefer newer construction with mid-to-upper assessed value.
    scored = []
    for row in parcels:
        yr = _to_int(row.get("year_of_construction")) or 0
        val = _to_float(row.get("assessed_value")) or 0
        if yr >= 1985 and 600_000 <= val <= 1_500_000:
            scored.append((yr, val, row))
    if scored:
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return scored[0][2]
    return parcels[0]


def _parcel_to_subject(row: dict[str, Any]) -> dict[str, Any]:
    lat, lon = _centroid(row.get("multipolygon") or {})
    community = _title_case_community(str(row.get("comm_name", "")))
    assessed = _to_float(row.get("assessed_value")) or 0
    year_built = _to_int(row.get("year_of_construction")) or 1990
    lot_sqft = _to_int(row.get("land_size_sf")) or 5000

    nb = config.NEIGHBORHOODS.get(community, {})
    base_ppsf = nb.get("base_ppsf", 450)
    gla_sqft = max(1200, min(3500, int(assessed / base_ppsf * 0.88)))
    bedrooms = 4 if gla_sqft >= 2000 else 3
    bathrooms = 3.0 if gla_sqft >= 2000 else 2.5

    return {
        "id": f"SUBJECT-{row.get('roll_number', 'OPEN')}",
        "address": str(row.get("address", "")).title(),
        "city": "Calgary",
        "neighborhood": community,
        "lat": lat or nb.get("lat", 51.05),
        "lon": lon or nb.get("lon", -114.07),
        "property_type": "Detached",
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "gla_sqft": gla_sqft,
        "lot_size_sqft": lot_sqft,
        "year_built": year_built,
        "roll_number": str(row.get("roll_number", "")),
        "assessed_value": assessed,
        "land_use_designation": row.get("land_use_designation"),
    }


def _assessment_row(subject: dict[str, Any], parcel: dict[str, Any]) -> dict[str, Any]:
    assessed = _to_int(parcel.get("assessed_value")) or 0
    return {
        "roll_number": str(parcel.get("roll_number", "")),
        "roll_year": str(parcel.get("roll_year", "")),
        "address": subject["address"],
        "property_class": parcel.get("assessment_class"),
        "property_class_description": parcel.get("assessment_class_description"),
        "property_type": "Detached",
        "land_use": parcel.get("land_use_designation"),
        "sub_property_use": parcel.get("sub_property_use"),
        "gla_sqft": subject["gla_sqft"],
        "living_area": subject["gla_sqft"],
        "lot_size_sqft": subject["lot_size_sqft"],
        "year_built": subject["year_built"],
        "assessed_value": assessed,
        "assessment_year": _to_int(parcel.get("roll_year")) or config.VALUATION_DATE.year,
        "comm_name": parcel.get("comm_name"),
        "comm_code": parcel.get("comm_code"),
        "source": "Open Calgary — Current Year Property Assessments (4bsw-nn7w)",
        "source_url": "https://data.calgary.ca/Government/Current-Year-Property-Assessments-Parcel-/4bsw-nn7w",
    }


def _fetch_permits(community: str, *, limit: int = 100) -> list[dict[str, Any]]:
    rows = _socrata_get(_PERMITS_API, {
        "communityname": community.upper(),
        "$order": "issueddate DESC",
        "$limit": str(limit),
    })
    out = []
    for r in rows:
        out.append({
            "permit_number": r.get("permitnum"),
            "status": r.get("statuscurrent"),
            "applied_date": (r.get("applieddate") or "")[:10],
            "issued_date": (r.get("issueddate") or "")[:10],
            "completed_date": (r.get("completeddate") or "")[:10] or None,
            "permit_type": r.get("permittype"),
            "permit_class": r.get("permitclass"),
            "work_class": r.get("workclass"),
            "description": r.get("description"),
            "est_project_cost": _to_float(r.get("estprojectcost")),
            "address": r.get("originaladdress"),
            "community": r.get("communityname"),
            "latitude": _to_float(r.get("latitude")),
            "longitude": _to_float(r.get("longitude")),
            "source": "Open Calgary — Building Permits (c2es-76ed)",
        })
    return out


def _geojson_feature(parcel: dict[str, Any], subject: dict[str, Any],
                     *, label: str, extra: dict) -> dict[str, Any]:
    geom = parcel.get("multipolygon")
    if not geom:
        geom = {"type": "Point", "coordinates": [subject["lon"], subject["lat"]]}
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "label": label,
                "address": subject["address"],
                "roll_number": parcel.get("roll_number"),
                "community": parcel.get("comm_name"),
                "land_use_designation": parcel.get("land_use_designation"),
                **extra,
            },
            "geometry": geom,
        }],
    }


def _nearby_synthetic_comps(subject: dict[str, Any]) -> pd.DataFrame:
    if not os.path.exists(_COMPS):
        raise FileNotFoundError(f"Run `python data/generate_data.py` first — missing {_COMPS}")
    df = pd.read_csv(_COMPS)
    df["sale_date"] = df["sale_date"].astype(str)
    df["distance_km"] = df.apply(
        lambda r: haversine_km(subject["lat"], subject["lon"], r["lat"], r["lon"]), axis=1
    )
    near = df[
        (df["property_type"] == "Detached")
        & (df["city"] == subject["city"])
        & (df["distance_km"] <= 4.0)
    ].sort_values("distance_km")
    band = near[near["gla_sqft"].between(subject["gla_sqft"] * 0.65, subject["gla_sqft"] * 1.35)]
    return (band if len(band) >= 15 else near).head(40).drop(columns=["distance_km"]).reset_index(drop=True)


def download_case(
    out_dir: str,
    *,
    community: str = DEFAULT_COMMUNITY,
    address: Optional[str] = DEFAULT_ADDRESS,
) -> None:
    case = os.path.join(_ROOT, out_dir) if not os.path.isabs(out_dir) else out_dir
    L = config.CASE_LAYOUT
    comm_upper = community.upper()

    print(f"Fetching Open Calgary assessments for {comm_upper} ...")
    parcels = _fetch_community_parcels(comm_upper)
    if not parcels:
        raise RuntimeError(f"No detached parcels found for community '{community}'")

    parcel = _pick_subject(parcels, address)
    subject = _parcel_to_subject(parcel)
    print(f"  subject: {subject['address']} | assessed ${subject['assessed_value']:,.0f} | built {subject['year_built']}")

    print("Fetching building permits ...")
    permits = _fetch_permits(comm_upper)
    print(f"  permits: {len(permits)}")

    comps = _nearby_synthetic_comps(subject)
    print(f"  synthetic comps (MLS substitute): {len(comps)}")

    def p(section: str, *parts: str) -> str:
        return os.path.join(case, L[section]["dir"], *parts)

    provenance = (
        "Downloaded from Open Calgary (data.calgary.ca) on case build. "
        "Comparable sale rows are synthetic (data/comps.csv) — MLS/Pillar 9 "
        "data requires a licensed feed."
    )

    # 00 assignment
    _touch(
        p("assignment", "effective_date.json"),
        json.dumps({
            "effective_date": config.VALUATION_DATE.isoformat(),
            "report_date": config.VALUATION_DATE.isoformat(),
            "client": "KV Capital Credit",
            "borrower": "Open Data Subject (Calgary Assessment)",
            "intended_use": "Mortgage financing / collateral valuation",
            "case_provenance": provenance,
            "subject_roll_number": subject.get("roll_number"),
        }, indent=2),
    )
    for key, fn in L["assignment"]["documents"].items():
        _placeholder(p("assignment", fn), key, note="Lender docs not in Open Calgary; placeholder.")

    # 01 subject
    _write_csv(p("subject", "subject_listing.csv"), [subject])
    for key, fn in L["subject"]["documents"].items():
        _placeholder(p("subject", fn), key, note="MLS/listing export not publicly available.")
    for photo in ("exterior_front", "exterior_rear", "kitchen", "living_room",
                  "bedrooms", "bathrooms", "basement", "garage"):
        _placeholder(p("subject", "subject_photos", f"{photo}.jpg"), "subject photo")

    # 02 legal/title
    for key, fn in L["legal_title"]["documents"].items():
        _placeholder(p("legal_title", fn), key, note="SPIN2 title/survey requires registry purchase.")
    _placeholder(p("legal_title", "encumbrances", "easement_001.pdf"), "easement")
    _placeholder(p("legal_title", "encumbrances", "restrictive_covenant_001.pdf"), "covenant")

    # 03 assessment/tax — real Open Calgary row
    _write_csv(p("assessment_tax", "assessment_open_data.csv"), [_assessment_row(subject, parcel)])
    for key, fn in L["assessment_tax"]["documents"].items():
        _placeholder(p("assessment_tax", fn), key,
                      note=f"See live assessment at https://assessment.calgary.ca/ for roll {subject.get('roll_number')}.")

    # 04 zoning — real parcel geometry from assessment multipolygon
    land_use = str(parcel.get("land_use_designation") or "R-C1")
    for key, extra in (
        ("land_use_district", {"district": land_use, "use": "Single detached (R110)"}),
        ("parcel_boundary", {"parcel_id": parcel.get("roll_number")}),
        ("community_boundary", {"community": comm_upper}),
        ("overlays", {"overlay": "none noted in Open Calgary assessment feed"}),
    ):
        _touch(p("zoning", f"{key}.geojson"),
               json.dumps(_geojson_feature(parcel, subject, label=key, extra=extra), indent=2))
    _touch(
        p("zoning", "zoning_hbu_notes.md"),
        f"# Zoning / HBU notes — {subject['address']}\n\n"
        f"**Source:** Open Calgary Current Year Property Assessments (dataset `4bsw-nn7w`)\n\n"
        f"- Community: {comm_upper}\n"
        f"- Land use designation: `{land_use}`\n"
        f"- Sub property use: `{parcel.get('sub_property_use')}` (R110 = single detached)\n"
        f"- Assessed value (2026 roll): ${subject['assessed_value']:,.0f}\n\n"
        "Present use as a single-family detached dwelling is consistent with "
        "the recorded land use. Highest and best use is continued residential.\n",
    )
    for key, fn in L["zoning"]["documents"].items():
        _placeholder(p("zoning", fn), key)

    # 05 permits — real Open Calgary permits (community-scoped)
    building = [r for r in permits if (r.get("permit_type") or "").startswith("Residential")
                or (r.get("permit_class") or "").startswith("1")]
    _write_csv(p("permits", "building_permits.csv"), building[:40] or permits[:20])
    _write_csv(p("permits", "development_permits.csv"), permits[:15])
    _write_csv(p("permits", "survey_measurements.csv"), [
        {"label": "GLA above grade (estimated)", "value_sqft": subject["gla_sqft"],
         "note": "Estimated from assessed value; not in Open Calgary CSV"},
        {"label": "Lot area (Open Calgary land_size_sf)", "value_sqft": subject["lot_size_sqft"],
         "source": "4bsw-nn7w"},
    ])
    for fn in ("garage_permit.pdf", "basement_permit.pdf", "addition_permit.pdf"):
        _placeholder(p("permits", "permit_documents", fn), "permit document")

    # 06 comparables — synthetic sales near subject (MLS not public)
    os.makedirs(os.path.join(case, L["comparables"]["dir"]), exist_ok=True)
    sold = comps.head(20)
    actives = comps.iloc[20:25] if len(comps) > 20 else comps.head(3)
    pending = comps.iloc[25:30] if len(comps) > 25 else comps.head(3)
    rejected = comps.iloc[30:34] if len(comps) > 30 else comps.head(3)
    sold.to_csv(p("comparables", "sold_comps_raw.csv"), index=False)
    actives.to_csv(p("comparables", "active_listings_raw.csv"), index=False)
    pending.to_csv(p("comparables", "conditional_pending_raw.csv"), index=False)
    rejected.to_csv(p("comparables", "rejected_comps.csv"), index=False)
    sold.head(8).to_csv(p("comparables", "verified_comps.csv"), index=False)
    for i, _ in enumerate(sold["id"].head(3)):
        _placeholder(p("comparables", "comp_photos", f"comp_{i+1:03d}_front.jpg"), "comp photo")
        _placeholder(p("comparables", "comp_listing_sheets", f"comp_{i+1:03d}_listing.pdf"), "comp listing")

    # 07 market context (synthetic CREB-style stubs — CREB reports are member-only)
    _write_csv(p("market", "price_index.csv"), [
        {"month": "2025-12", "index": 198.4, "community": comm_upper},
        {"month": "2026-01", "index": 199.1, "community": comm_upper},
        {"month": "2026-05", "index": 203.7, "community": comm_upper},
    ])
    _write_csv(p("market", "absorption_inventory.csv"), [
        {"month": "2026-05", "community": comm_upper,
         "active_listings": 98, "sales": 41, "months_of_supply": 2.4},
    ])
    _write_csv(p("market", "days_on_market_stats.csv"), [
        {"month": "2026-05", "community": comm_upper, "median_dom": 19, "avg_dom": 26},
    ])
    for key, fn in L["market"]["documents"].items():
        _placeholder(p("market", fn), key, note="CREB member report; placeholder.")

    os.makedirs(os.path.join(case, L["workflow_outputs"]["dir"]), exist_ok=True)
    os.makedirs(os.path.join(case, L["final_package"]["dir"]), exist_ok=True)

    manifest = ds.case_manifest()
    manifest["case_provenance"] = provenance
    manifest["open_calgary_subject"] = {
        "roll_number": subject.get("roll_number"),
        "address": subject["address"],
        "community": comm_upper,
        "assessed_value": subject["assessed_value"],
        "downloaded_from": [
            "https://data.calgary.ca/resource/4bsw-nn7w.json",
            "https://data.calgary.ca/resource/c2es-76ed.json",
        ],
    }
    _touch(os.path.join(case, "data_source_manifest.json"), json.dumps(manifest, indent=2))

    print(f"\nWrote Open Calgary case to {case}")
    print(f"  community parcels fetched: {len(parcels)}")
    print(f"  sold comps (synthetic): {len(sold)} | permits: {len(permits)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download a Calgary valuation case from Open Data")
    ap.add_argument("--out", default="valuation_case_001", help="output directory name")
    ap.add_argument("--community", default=DEFAULT_COMMUNITY, help="Open Calgary comm_name")
    ap.add_argument("--address", default=DEFAULT_ADDRESS, help="subject address (uppercase match)")
    args = ap.parse_args()
    download_case(args.out, community=args.community, address=args.address or None)


if __name__ == "__main__":
    main()
