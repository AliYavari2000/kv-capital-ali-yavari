"""Node 9 - Adjustment Engine (deterministic / statistical, NOT LLM-driven).

Per the design, this node is deterministic: it applies market-supported
sales-comparison adjustments to each comp (time, type, GLA, beds/baths, lot,
age via ``valuation_math.compute_adjustments``), builds the adjustment grid,
checks for excessive gross/net adjustments, and runs a simple sensitivity test.
An LLM is NOT used here -- every number traces back to a coefficient in
``src/config.py``.

Tool surface implemented: time_adjustment_model, gla/lot/bed-bath/garage/
basement/condition/quality/location adjustment calculators (inside
``compute_adjustments``), gross_net_adjustment_checker, adjustment_grid_builder,
sensitivity_analyzer, append_evidence, raise_human_review.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from src import config, valuation_math as vm
from src.state import CompState

# Gross adjustment beyond this share of sale price is an appraisal concern.
_GROSS_FLAG = config.RISK["gross_adj_high"]


def _gross_net_checker(comps: list[dict[str, Any]]) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    for c in comps:
        gp = float(c.get("gross_adjustment_pct", 0.0))
        if gp >= _GROSS_FLAG:
            flags.append({
                "code": "excessive_adjustment", "severity": "medium",
                "message": f"Comp {c.get('id')} required {gp*100:.0f}% gross adjustment.",
            })
    return flags


def _adjustment_grid(comps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grid = []
    for c in comps:
        grid.append({
            "id": c.get("id"),
            "sale_price": c.get("sale_price"),
            "adjustments": {a["factor"]: a["amount"] for a in c.get("adjustments", [])},
            "net_adjustment": c.get("net_adjustment"),
            "adjusted_price": c.get("adjusted_price"),
            "gross_adjustment_pct": c.get("gross_adjustment_pct"),
        })
    return grid


def _sensitivity(comps: list[dict[str, Any]]) -> dict[str, Any]:
    """How the simple mean adjusted value moves if all adjustments scale +/-20%."""
    if not comps:
        return {}
    base = [float(c["adjusted_price"]) for c in comps]
    base_mean = sum(base) / len(base)
    scaled = {}
    for label, k in (("minus_20pct", 0.8), ("plus_20pct", 1.2)):
        vals = [float(c["sale_price"]) + k * float(c.get("net_adjustment", 0)) for c in comps]
        scaled[label] = round(sum(vals) / len(vals), -2)
    return {"base_mean": round(base_mean, -2), **scaled}


def adjustment_engine_node(state: CompState) -> dict[str, Any]:
    subject = state.get("subject", {})
    ranked = state.get("ranked_comps", [])

    adjusted: list[dict[str, Any]] = []
    for comp in ranked:
        result = vm.compute_adjustments(subject, comp)
        merged = dict(comp)
        merged.update(result)
        adjusted.append(merged)

    flags = _gross_net_checker(adjusted)
    grid = _adjustment_grid(adjusted)
    sensitivity = _sensitivity(adjusted)
    avg_gross = (sum(c["gross_adjustment_pct"] for c in adjusted) / len(adjusted)) if adjusted else 0.0

    evidence = state.get("evidence", []) + [{
        "node": "adjustment_engine",
        "source": "src/config.py coefficients",
        "detail": f"Applied deterministic sales-comparison adjustments to {len(adjusted)} comps.",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
    }]

    trace = state.get("trace", []) + [
        f"adjustment_engine (deterministic): {len(adjusted)} comps adjusted, mean gross "
        f"{avg_gross*100:.1f}%, excessive={len(flags)}, "
        f"sensitivity={sensitivity.get('minus_20pct')}..{sensitivity.get('plus_20pct')}"
    ]
    return {
        "adjusted_comps": adjusted,
        "ranked_comps": adjusted,
        "adjustment_grid": grid,
        "adjustment_flags": flags,
        "sensitivity": sensitivity,
        "evidence": evidence,
        "trace": trace,
    }
