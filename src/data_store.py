"""Comparable-sales store: loads the synthetic dataset and answers filtered
retrieval queries (geo radius, property type, recency, GLA band) with
progressive widening when too few candidates are found.

In production this layer would wrap an MLS / land-titles feed; the rest of the
agent is agnostic to where comps come from.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import pandas as pd

from src import config
from src.valuation_math import haversine_km, months_between

_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "comps.csv")


@functools.lru_cache(maxsize=1)
def load_comps(path: str = _DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Comps dataset not found at {path}. Run `python data/generate_data.py` first."
        )
    df = pd.read_csv(path)
    df["sale_date"] = df["sale_date"].astype(str)
    return df


def _rows_to_comps_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize case-CSV rows onto the canonical schema and build a DataFrame
    the retrieval/similarity code can consume."""
    from src import case_store

    normalized = case_store.normalize_comp_columns(rows)
    df = pd.DataFrame(normalized)
    if df.empty:
        return df
    if "sale_date" in df.columns:
        df["sale_date"] = df["sale_date"].astype(str)
    return df


@functools.lru_cache(maxsize=4)
def load_case_comps(case_dir: str) -> pd.DataFrame:
    """Load the sold comparables for a case from ``06_comparables/``.

    Uses ``sold_comps_raw.csv`` as the candidate pool, falling back to
    ``verified_comps.csv`` when the raw pool is empty. Columns are normalized to
    the canonical schema so the rest of the agent is source-agnostic.
    """
    from src import case_store

    case = case_store.load_case(case_dir)
    comps = case.comps()["data"]
    rows = comps.get("sold_comps_raw") or comps.get("verified_comps") or []
    if not isinstance(rows, list):
        rows = []
    df = _rows_to_comps_df(rows)
    if df.empty:
        raise ValueError(
            f"No comparable sales found in case {case_dir} "
            f"(06_comparables/sold_comps_raw.csv)."
        )
    return df


_FAR_AWAY_KM = 9999.0  # sentinel for comps missing coordinates (geocoding is a later tool)


def _distance_or_sentinel(subject: dict[str, Any], row: pd.Series) -> float:
    s_lat, s_lon = subject.get("lat"), subject.get("lon")
    r_lat, r_lon = row.get("lat"), row.get("lon")
    if None in (s_lat, s_lon, r_lat, r_lon) or pd.isna(r_lat) or pd.isna(r_lon):
        return _FAR_AWAY_KM
    return haversine_km(float(s_lat), float(s_lon), float(r_lat), float(r_lon))


def _annotate(df: pd.DataFrame, subject: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["distance_km"] = out.apply(lambda r: _distance_or_sentinel(subject, r), axis=1)
    out["months_ago"] = out["sale_date"].apply(lambda d: months_between(d, config.VALUATION_DATE))
    return out


def _apply_filters(
    df: pd.DataFrame,
    subject: dict[str, Any],
    *,
    radius_km: float,
    recency_months: float,
    gla_band: float,
    include_adjacent_types: bool,
    allowed_types: set[str] | None = None,
) -> pd.DataFrame:
    mask = (df["distance_km"] <= radius_km) & (df["months_ago"] <= recency_months)

    subj_gla = float(subject.get("gla_sqft") or 0)
    if subj_gla > 0:
        mask &= df["gla_sqft"].between(subj_gla * (1 - gla_band), subj_gla * (1 + gla_band))

    if allowed_types:
        # An explicit allow-list from the market-scope step takes precedence.
        allowed = set(allowed_types)
        if include_adjacent_types:
            order = config.PROPERTY_TYPES
            for t in list(allowed):
                if t in order:
                    i = order.index(t)
                    allowed |= {order[j] for j in (i - 1, i + 1) if 0 <= j < len(order)}
        mask &= df["property_type"].isin(allowed)
    else:
        subj_type = subject.get("property_type")
        if subj_type:
            if include_adjacent_types:
                # Allow neighboring price tiers (e.g. Townhouse <-> Semi-Detached).
                order = config.PROPERTY_TYPES
                if subj_type in order:
                    i = order.index(subj_type)
                    allowed = {order[j] for j in (i - 1, i, i + 1) if 0 <= j < len(order)}
                else:
                    allowed = {subj_type}
                mask &= df["property_type"].isin(allowed)
            else:
                mask &= df["property_type"] == subj_type

    return df[mask]


def retrieve_candidates(
    subject: dict[str, Any],
    scope: dict[str, Any] | None = None,
    *,
    case_dir: str | None = None,
) -> dict[str, Any]:
    """Return candidate comps for a subject plus metadata about the search.

    The initial search envelope comes from the market-scope step (``scope``) when
    provided, otherwise from ``config.RETRIEVAL``. Widens the radius / recency /
    GLA band step-by-step until at least ``min_candidates`` are found or the
    attempt budget is exhausted.

    When ``case_dir`` is given, comps are sourced from the case folder's
    ``06_comparables/`` instead of the synthetic dataset.
    """
    source_df = load_case_comps(case_dir) if case_dir else load_comps()
    df = _annotate(source_df, subject)
    p = config.RETRIEVAL
    scope = scope or {}

    radius = float(scope.get("radius_km", p["radius_km"]))
    recency = float(scope.get("recency_months", p["recency_months"]))
    gla_band = float(scope.get("gla_band", p["gla_band"]))
    allowed_types = set(scope.get("property_types") or []) or None
    include_adjacent = False

    attempts: list[dict[str, Any]] = []
    result = pd.DataFrame()
    for attempt in range(p["max_widen_attempts"] + 1):
        result = _apply_filters(
            df, subject,
            radius_km=radius, recency_months=recency,
            gla_band=gla_band, include_adjacent_types=include_adjacent,
            allowed_types=allowed_types,
        )
        attempts.append({
            "attempt": attempt,
            "radius_km": round(radius, 2),
            "recency_months": round(recency, 1),
            "gla_band": round(gla_band, 2),
            "include_adjacent_types": include_adjacent,
            "found": int(len(result)),
        })
        if len(result) >= p["min_candidates"]:
            break
        # Widen for the next attempt.
        radius *= p["widen_radius_factor"]
        recency += p["widen_recency_add_months"]
        gla_band += p["widen_gla_band_add"]
        if attempt >= 1:
            include_adjacent = True

    result = result.sort_values("distance_km").head(p["max_candidates"])
    candidates = result.round({"distance_km": 3, "months_ago": 2}).to_dict(orient="records")

    return {
        "candidates": candidates,
        "meta": {
            "attempts": attempts,
            "final_filters": attempts[-1],
            "widened": len(attempts) > 1,
            "candidate_count": len(candidates),
        },
    }


def _case_market_context(case_dir: str, *, max_each: int = 3) -> dict[str, Any] | None:
    """Build active + pending/conditional context from the case's real listing
    CSVs (``06_comparables/active_listings_raw.csv`` and
    ``conditional_pending_raw.csv``). Returns ``None`` if neither is present."""
    from src import case_store

    case = case_store.load_case(case_dir)
    comps = case.comps()["data"]
    active_rows = comps.get("active_listings_raw") or []
    pending_rows = comps.get("conditional_pending_raw") or []
    if not isinstance(active_rows, list):
        active_rows = []
    if not isinstance(pending_rows, list):
        pending_rows = []
    if not active_rows and not pending_rows:
        return None

    def _shape(rows: list[dict[str, Any]], status: str) -> list[dict[str, Any]]:
        out = []
        for c in case_store.normalize_comp_columns(rows)[:max_each]:
            out.append({
                "id": c.get("id"),
                "address": c.get("address"),
                "neighborhood": c.get("neighborhood"),
                "property_type": c.get("property_type"),
                "gla_sqft": c.get("gla_sqft"),
                "status": status,
                "list_price": c.get("sale_price"),
            })
        return out

    return {
        "active": _shape(active_rows, "Active"),
        "pending": _shape(pending_rows, "Pending/Conditional"),
        "note": "active/pending context from the case 06_comparables listing files",
    }


def market_context(
    candidates: list[dict[str, Any]],
    *,
    max_each: int = 3,
    case_dir: str | None = None,
) -> dict[str, Any]:
    """Derive *active* and *pending/conditional* listing context.

    With a ``case_dir``, the real listing CSVs from the case folder are used.
    Otherwise (synthetic dataset, which carries only closed sales) the most
    recent nearby sales are used as proxies for current supply: list prices are
    the closed price marked up to the valuation date, with actives priced a touch
    above pendings. Clearly synthetic, but enough to bracket the market.
    """
    if case_dir:
        ctx = _case_market_context(case_dir, max_each=max_each)
        if ctx is not None:
            return ctx

    if not candidates:
        return {"active": [], "pending": [], "note": "no candidate pool"}

    pool = sorted(candidates, key=lambda c: (c.get("months_ago", 99), c.get("distance_km", 99)))
    actives, pendings = [], []
    for i, c in enumerate(pool[: max_each * 2]):
        sale = float(c.get("sale_price", 0) or 0)
        months = float(c.get("months_ago", 0) or 0)
        trended = sale * (1.0 + config.MONTHLY_APPRECIATION) ** months
        row = {
            "id": c.get("id"),
            "address": c.get("address"),
            "neighborhood": c.get("neighborhood"),
            "property_type": c.get("property_type"),
            "gla_sqft": c.get("gla_sqft"),
            "distance_km": c.get("distance_km"),
        }
        if i % 2 == 0 and len(actives) < max_each:
            row["status"] = "Active"
            row["list_price"] = int(round(trended * 1.03, -3))
            actives.append(row)
        elif len(pendings) < max_each:
            row["status"] = "Pending/Conditional"
            row["list_price"] = int(round(trended * 1.005, -3))
            pendings.append(row)

    return {
        "active": actives,
        "pending": pendings,
        "note": "synthetic active/pending context derived from recent nearby sales",
    }


def dataset_summary() -> dict[str, Any]:
    df = load_comps()
    return {
        "records": int(len(df)),
        "cities": df["city"].value_counts().to_dict(),
        "types": df["property_type"].value_counts().to_dict(),
        "median_price": int(df["sale_price"].median()),
    }
