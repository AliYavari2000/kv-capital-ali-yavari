"""Node 9 - Adjustment Engine (``AdjustmentEngineNode``).

Applies market-supported, sales-comparison adjustments to each verified,
normalized comp: time/market, property type, GLA (size), bedrooms, bathrooms,
lot, and effective age, producing an itemized breakdown, a net and gross
adjustment, and an adjusted sale price stated in effective-date dollars.

(Condition / garage / basement enter the grid as neutral when no differential
feature data is available; the seam is in ``valuation_math.compute_adjustments``.)

Tools: paired-sales model, hedonic coefficients, rules.
"""

from __future__ import annotations

from typing import Any

from src import valuation_math as vm
from src.state import CompState


def adjustment_engine_node(state: CompState) -> dict[str, Any]:
    subject = state.get("subject", {})
    ranked = state.get("ranked_comps", [])

    adjusted: list[dict[str, Any]] = []
    for comp in ranked:
        result = vm.compute_adjustments(subject, comp)
        merged = dict(comp)
        merged.update(result)
        adjusted.append(merged)

    if adjusted:
        avg_gross = sum(c["gross_adjustment_pct"] for c in adjusted) / len(adjusted)
    else:
        avg_gross = 0.0

    trace = state.get("trace", []) + [
        f"adjustment_engine: {len(adjusted)} comps adjusted, mean gross adjustment "
        f"{avg_gross*100:.1f}% of sale price"
    ]
    return {"adjusted_comps": adjusted, "ranked_comps": adjusted, "trace": trace}
