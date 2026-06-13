"""Synthesize a realistic Alberta residential sales dataset.

The price of every record is generated from the SAME transparent model the
agent uses to value subjects (see ``src/config.py``):

    value_today = gla * base_ppsf * type_multiplier
                + (beds  - typical_beds)  * BED_VALUE
                + (baths - typical_baths) * BATH_VALUE
                + (lot   - typical_lot)   * LOT_PPSF        (lot-bearing types)
                - min(age, cap)           * AGE_VALUE_PER_YEAR
    sale_price  = value_today * (1 + monthly_appreciation) ** (-months_ago)
                * lognormal_noise

Because adjustments are baked in here, the agent's recovered valuation is
checkable against ground truth.

Usage:
    python data/generate_data.py            # writes data/comps.csv
    python data/generate_data.py --n 4000   # custom record count
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

# Allow running as a plain script: make the project root importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402

RNG_SEED = 42

STREET_NAMES = [
    "Maple", "Spruce", "Riverside", "Crowchild", "Elbow", "Bow", "Memorial",
    "Whyte", "Jasper", "Saddleback", "Aspen", "Willow", "Cedar", "Hawthorn",
    "Sunalta", "Prospect", "Township", "Heritage", "Glenmore", "Macleod",
]
STREET_SUFFIX = ["St", "Ave", "Rd", "Dr", "Cres", "Way", "Blvd", "Gate"]
CALGARY_QUADRANTS = ["NW", "NE", "SW", "SE"]

# GLA (gross living area, sqft) ranges by type.
GLA_RANGE = {
    "Detached": (1400, 3200),
    "Semi-Detached": (1100, 2000),
    "Townhouse": (900, 1700),
    "Condo": (550, 1300),
}
# Relative frequency of each property type in the synthetic market.
TYPE_WEIGHTS = {
    "Detached": 0.45,
    "Semi-Detached": 0.15,
    "Townhouse": 0.18,
    "Condo": 0.22,
}
TYPICAL_LOT = {"Detached": 4500, "Semi-Detached": 3000}


def _months_between(start: date, end: date) -> float:
    return (end.year - start.year) * 12 + (end.month - start.month) + (end.day - start.day) / 30.0


def typical_beds(gla: float) -> int:
    return int(np.clip(round(1 + gla / 700.0), 1, 6))


def typical_baths(gla: float) -> float:
    return float(np.clip(round((1 + gla / 900.0) * 2) / 2.0, 1.0, 4.5))


def value_today(
    *,
    gla: float,
    base_ppsf: float,
    ptype: str,
    beds: int,
    baths: float,
    lot: float,
    age: int,
) -> float:
    """Contributory-value model in valuation-date dollars (pre-noise)."""
    type_mult = config.TYPE_PPSF_MULTIPLIER[ptype]
    val = gla * base_ppsf * type_mult
    val += (beds - typical_beds(gla)) * config.BED_VALUE
    val += (baths - typical_baths(gla)) * config.BATH_VALUE
    if config.TYPE_HAS_LOT[ptype]:
        val += (lot - TYPICAL_LOT[ptype]) * config.LOT_PPSF
    val -= min(age, config.MAX_AGE_FOR_DEPRECIATION) * config.AGE_VALUE_PER_YEAR
    return val


def generate_dataframe(n: int = 2500, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    neighborhoods = list(config.NEIGHBORHOODS.keys())
    types = list(TYPE_WEIGHTS.keys())
    type_p = np.array([TYPE_WEIGHTS[t] for t in types])
    type_p = type_p / type_p.sum()

    earliest = config.VALUATION_DATE - timedelta(days=int(config.SALES_HISTORY_MONTHS * 30.4))
    horizon_days = (config.VALUATION_DATE - earliest).days

    rows = []
    for i in range(n):
        nb = rng.choice(neighborhoods)
        meta = config.NEIGHBORHOODS[nb]
        ptype = rng.choice(types, p=type_p)

        lo, hi = GLA_RANGE[ptype]
        gla = int(rng.integers(lo, hi + 1))

        beds = int(np.clip(typical_beds(gla) + rng.choice([-1, 0, 0, 0, 1]), 1, 7))
        baths = float(np.clip(typical_baths(gla) + rng.choice([-1.0, -0.5, 0.0, 0.0, 0.5, 1.0]), 1.0, 5.0))

        if config.TYPE_HAS_LOT[ptype]:
            lot_lo, lot_hi = (2500, 7500) if ptype == "Detached" else (2000, 4200)
            lot = int(rng.integers(lot_lo, lot_hi + 1))
        else:
            lot = 0

        year_built = int(rng.integers(1955, config.VALUATION_DATE.year + 1))
        age = config.VALUATION_DATE.year - year_built

        # Jitter the centroid so comps are not all stacked on one point.
        lat = float(meta["lat"] + rng.normal(0, 0.012))
        lon = float(meta["lon"] + rng.normal(0, 0.016))

        sale_date = earliest + timedelta(days=int(rng.integers(0, horizon_days + 1)))
        months_ago = _months_between(sale_date, config.VALUATION_DATE)

        v_today = value_today(
            gla=gla, base_ppsf=meta["base_ppsf"], ptype=ptype,
            beds=beds, baths=baths, lot=lot, age=age,
        )
        trend = (1.0 + config.MONTHLY_APPRECIATION) ** (-months_ago)
        noise = float(rng.lognormal(mean=0.0, sigma=0.06))
        price = max(80_000.0, v_today * trend * noise)

        # Address
        if meta["city"] == "Calgary":
            quad = rng.choice(CALGARY_QUADRANTS)
            address = f"{int(rng.integers(100, 9999))} {rng.choice(STREET_NAMES)} {rng.choice(STREET_SUFFIX)} {quad}"
        else:
            address = f"{int(rng.integers(100, 19999))} {rng.choice(STREET_NAMES)} {rng.choice(STREET_SUFFIX)} NW"

        rows.append({
            "id": f"AB-{i:05d}",
            "address": address,
            "city": meta["city"],
            "neighborhood": nb,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "property_type": ptype,
            "bedrooms": beds,
            "bathrooms": baths,
            "gla_sqft": gla,
            "lot_size_sqft": lot,
            "year_built": year_built,
            "sale_date": sale_date.isoformat(),
            "sale_price": int(round(price, -2)),
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Alberta comps")
    parser.add_argument("--n", type=int, default=2500, help="number of records")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "comps.csv"),
    )
    args = parser.parse_args()

    df = generate_dataframe(n=args.n, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} records to {args.out}")
    print(f"Cities: {df['city'].value_counts().to_dict()}")
    print(f"Types:  {df['property_type'].value_counts().to_dict()}")
    print(f"Price:  ${df['sale_price'].min():,} - ${df['sale_price'].max():,} "
          f"(median ${int(df['sale_price'].median()):,})")


if __name__ == "__main__":
    main()
