"""Node 10 - Bracketing and Reconciliation.

The value math is deterministic: weighting follows appraisal practice (the most
similar, least-adjusted comps drive the estimate), and the range, median /
trimmed-mean cross-checks, bracketing test, and confidence/risk view are all
computed, not generated. The LLM then EXPLAINS the result (it does not compute
it) via ``llm.write_reconciliation``.

No deterministic fallback for the explanation: requires an LLM.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from src import config, llm
from src.state import CompState

# How sharply gross adjustment penalizes a comp's weight.
_ADJ_PENALTY = 4.0


def _flag(code: str, severity: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def _reconcile(comps: list[dict[str, Any]], subject: dict[str, Any]) -> dict[str, Any]:
    weights = []
    for c in comps:
        sim = float(c.get("similarity", 0.0))
        gross_pct = float(c.get("gross_adjustment_pct", 0.0))
        w = (sim ** 2) / (1.0 + _ADJ_PENALTY * gross_pct)
        weights.append(max(w, 1e-6))
    total_w = sum(weights)

    for c, w in zip(comps, weights):
        c["weight"] = round(w / total_w, 4)

    prices = [float(c["adjusted_price"]) for c in comps]
    point = sum(w * p for w, p in zip(weights, prices)) / total_w
    variance = sum(w * (p - point) ** 2 for w, p in zip(weights, prices)) / total_w
    spread = math.sqrt(variance)

    lo = max(min(prices), point - spread)
    hi = min(max(prices), point + spread)
    if hi <= lo:  # degenerate (e.g. one comp): fall back to +/-3%
        lo, hi = point * 0.97, point * 1.03

    gla = float(subject.get("gla_sqft") or 0)
    implied_ppsf = point / gla if gla else 0.0

    # Cross-checks: median + trimmed mean of adjusted prices.
    median_adj = statistics.median(prices)
    if len(prices) >= 4:
        s = sorted(prices)
        trimmed = statistics.fmean(s[1:-1])
    else:
        trimmed = statistics.fmean(prices)

    # Bracketing: is the estimate bracketed by the adjusted comps?
    bracketed = min(prices) <= point <= max(prices)

    return {
        "point_estimate": round(point, -2),
        "mid": round(point, -2),
        "low": round(lo, -2),
        "high": round(hi, -2),
        "median_adjusted": round(median_adj, -2),
        "trimmed_mean_adjusted": round(trimmed, -2),
        "bracketed": bool(bracketed),
        "implied_ppsf": round(implied_ppsf, 0),
        "method": "similarity_weighted_sales_comparison",
        "comp_count": len(comps),
    }


def reconciliation_node(state: CompState) -> dict[str, Any]:
    if not llm.llm_available():
        raise llm.LLMUnavailableError(
            "Reconciliation requires OPENAI_API_KEY to write the explanation."
        )
    comps = state.get("adjusted_comps", []) or state.get("ranked_comps", [])
    subject = state.get("subject", {})
    dq = state.get("data_quality", {})
    r = config.RISK

    if not comps:
        valuation = {"point_estimate": 0.0, "mid": 0.0, "low": 0.0, "high": 0.0,
                     "implied_ppsf": 0.0, "method": "no_comps", "comp_count": 0}
        risk = {"confidence": "Low", "flags": [_flag("no_comps", "high", "No comparable sales were found.")],
                "requires_human_review": True, "metrics": {"comp_count": 0}}
        trace = state.get("trace", []) + ["reconciliation: no comps available -> human sign-off required"]
        return {"valuation": valuation, "risk": risk, "trace": trace}

    valuation = _reconcile(comps, subject)

    # --- Risk view -------------------------------------------------------
    flags: list[dict[str, str]] = []
    adj_prices = [float(c["adjusted_price"]) for c in comps]
    mean_price = statistics.fmean(adj_prices) if adj_prices else 0.0
    cov = (statistics.pstdev(adj_prices) / mean_price) if len(adj_prices) > 1 and mean_price else 0.0
    mean_gross = statistics.fmean([float(c.get("gross_adjustment_pct", 0)) for c in comps]) if comps else 0.0
    median_months = statistics.median([float(c.get("months_ago", 0)) for c in comps]) if comps else 0.0
    median_dist = statistics.median([float(c.get("distance_km", 0)) for c in comps]) if comps else 0.0
    dq_score = float(dq.get("score", 1.0))

    if len(comps) < r["min_comps"]:
        flags.append(_flag("thin_comps", "high", f"Only {len(comps)} comparable(s) found (min {r['min_comps']})."))

    if cov >= r["cov_high"]:
        flags.append(_flag("high_dispersion", "high", f"Adjusted values vary widely (CoV {cov*100:.1f}%)."))
    elif cov >= r["cov_elevated"]:
        flags.append(_flag("dispersion", "medium", f"Moderate spread in adjusted values (CoV {cov*100:.1f}%)."))

    if mean_gross >= r["gross_adj_high"]:
        flags.append(_flag("large_adjustments", "high", f"Comps required large adjustments (mean {mean_gross*100:.1f}% of price)."))
    elif mean_gross >= r["gross_adj_elevated"]:
        flags.append(_flag("elevated_adjustments", "medium", f"Comps required notable adjustments (mean {mean_gross*100:.1f}%)."))

    if median_months > r["stale_months"]:
        flags.append(_flag("stale_comps", "medium", f"Median comp age is {median_months:.1f} months."))

    if median_dist > r["far_km"]:
        flags.append(_flag("distant_comps", "medium", f"Median comp distance is {median_dist:.1f} km."))

    if not dq.get("passed", True):
        flags.append(_flag("data_quality", "high", "Subject is missing critical fields or failed sanity checks."))
    elif dq_score < r["data_quality_min"]:
        flags.append(_flag("weak_inputs", "medium", f"Subject data is incomplete (quality {dq_score:.2f})."))

    if state.get("retrieval_meta", {}).get("widened"):
        flags.append(_flag("widened_search", "medium", "Search criteria were widened to find enough comps."))

    # Fold in upstream legal/title, zoning, and verification findings.
    flags.extend(state.get("legal_title", {}).get("flags", []))
    flags.extend(state.get("zoning", {}).get("flags", []))
    flags.extend(state.get("verification", {}).get("flags", []))

    highs = sum(1 for f in flags if f["severity"] == "high")
    mediums = sum(1 for f in flags if f["severity"] == "medium")
    if highs >= 1:
        confidence = "Low"
    elif mediums >= 1:
        confidence = "Medium"
    else:
        confidence = "High"

    requires_human = confidence in config.HUMAN_REVIEW_ON or not dq.get("passed", True)

    risk = {
        "confidence": confidence,
        "flags": flags,
        "requires_human_review": requires_human,
        "metrics": {
            "comp_count": len(comps),
            "coefficient_of_variation": round(cov, 4),
            "mean_gross_adjustment_pct": round(mean_gross, 4),
            "median_comp_age_months": round(median_months, 1),
            "median_comp_distance_km": round(median_dist, 2),
            "data_quality_score": dq_score,
        },
    }

    # LLM explains the deterministic result (does not compute it).
    narrative = llm.write_reconciliation({
        "valuation": valuation,
        "confidence": confidence,
        "comps": [{"id": c.get("id"), "adjusted_price": c.get("adjusted_price"),
                   "weight": c.get("weight"), "similarity": c.get("similarity"),
                   "gross_adjustment_pct": c.get("gross_adjustment_pct")} for c in comps],
        "bracketed": valuation.get("bracketed"),
        "flags": [f["message"] for f in flags],
    })
    valuation["reconciliation_narrative"] = narrative

    trace = state.get("trace", []) + [
        f"reconciliation (math + LLM explanation): ${valuation['point_estimate']:,.0f} "
        f"(range ${valuation['low']:,.0f}-${valuation['high']:,.0f}, "
        f"${valuation['implied_ppsf']:,.0f}/sqft) from {len(comps)} comps; "
        f"bracketed={valuation.get('bracketed')}, confidence={confidence}, "
        f"flags={highs}H/{mediums}M, "
        f"human_sign_off={'required' if requires_human else 'not required'}"
    ]
    return {"valuation": valuation, "risk": risk,
            "adjusted_comps": comps, "ranked_comps": comps, "trace": trace}
