"""Node 11 - Report Writer (``ReportWriterNode``).

Documents the report, assumptions, and workfile. Produces the deliverable in two
forms: a structured JSON object (systems/audit) and a human-readable markdown
appraisal memo: assignment + effective date, subject, legal/title, zoning & HBU,
market scope, the comp grid with itemized adjustments, active/pending context,
the reconciled value, risk, assumptions & limiting conditions, citations, and an
LLM-written rationale grounded only in the computed numbers.

Tools: report generator, PDF/DOCX export.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src import config, llm
from src.state import CompState


def _fmt_money(v: Any) -> str:
    try:
        return f"${float(v):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def _assignment_block(a: dict[str, Any]) -> str:
    return "\n".join([
        f"- Client: {a.get('client', 'n/a')}",
        f"- Borrower: {a.get('borrower', 'n/a')}",
        f"- Intended use: {a.get('intended_use', 'n/a')}",
        f"- Effective (valuation) date: {a.get('effective_date', 'n/a')}",
        f"- Report date: {a.get('report_date', 'n/a')}",
    ])


def _subject_block(s: dict[str, Any]) -> str:
    lines = [
        f"- Address: {s.get('address', 'n/a')}",
        f"- Location: {s.get('neighborhood', 'n/a')}, {s.get('city', 'n/a')}",
        f"- Type: {s.get('property_type', 'n/a')}",
        f"- Beds/Baths: {s.get('bedrooms', 'n/a')} / {s.get('bathrooms', 'n/a')}",
        f"- GLA: {s.get('gla_sqft', 'n/a')} sqft"
        + (f" | Lot: {s.get('lot_size_sqft')} sqft" if s.get("lot_size_sqft") else ""),
        f"- Year built: {s.get('year_built', 'n/a')}"
        + (f" (age {s.get('property_age')})" if s.get("property_age") is not None else ""),
        f"- Condition: {s.get('condition', 'n/a')}",
    ]
    return "\n".join(lines)


def _legal_block(lt: dict[str, Any]) -> str:
    if not lt:
        return "_Not assessed._"
    flags = lt.get("flags", [])
    flag_txt = "\n".join(f"  - [{f['severity'].upper()}] {f['message']}" for f in flags) if flags else "  - None"
    return "\n".join([
        f"- Address confirmed: {lt.get('address_confirmed')}",
        f"- Legal description: {lt.get('legal_description', 'n/a')}",
        f"- Parcel ID (LINC): {lt.get('parcel_id', 'n/a')}",
        f"- Title status: {lt.get('title_status', 'n/a')}",
        f"- Registered owner: {lt.get('registered_owner', 'n/a')}",
        "- Title flags:",
        flag_txt,
    ])


def _zoning_block(z: dict[str, Any]) -> str:
    if not z:
        return "_Not assessed._"
    flags = z.get("flags", [])
    flag_txt = "\n".join(f"  - [{f['severity'].upper()}] {f['message']}" for f in flags) if flags else "  - None"
    return "\n".join([
        f"- Zoning: {z.get('zoning_code', 'n/a')}",
        f"- Permitted use: {z.get('permitted_use', 'n/a')}",
        f"- Conforming: {z.get('conforming')}",
        f"- Highest & best use: {z.get('highest_and_best_use', 'n/a')}",
        "- Zoning flags:",
        flag_txt,
    ])


def _scope_block(ms: dict[str, Any]) -> str:
    if not ms:
        return "_Not assessed._"
    return "\n".join([
        f"- Radius: {ms.get('radius_km')} km",
        f"- Recency window: {ms.get('recency_months')} months",
        f"- GLA band: +/-{int(float(ms.get('gla_band', 0))*100)}%",
        f"- Comparable types: {ms.get('property_types')}",
        f"- Rationale: {ms.get('rationale', 'n/a')}",
    ])


def _comps_table(comps: list[dict[str, Any]]) -> str:
    if not comps:
        return "_No comparable sales were available._"
    header = (
        "| Comp | Location | Type | Bd/Ba | GLA | Sold | Dist (km) | Age (mo) | Sale | Net Adj | Adjusted | Sim | Wt | Verified |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for c in comps:
        rows.append(
            f"| {c.get('id')} | {c.get('neighborhood')} | {c.get('property_type')} "
            f"| {c.get('bedrooms')}/{c.get('bathrooms')} | {c.get('gla_sqft')} "
            f"| {str(c.get('sale_date'))[:10]} | {c.get('distance_km'):.2f} "
            f"| {c.get('months_ago'):.1f} | {_fmt_money(c.get('sale_price'))} "
            f"| {_fmt_money(c.get('net_adjustment'))} | {_fmt_money(c.get('adjusted_price'))} "
            f"| {c.get('similarity', 0):.2f} | {c.get('weight', 0):.2f} "
            f"| {'yes' if c.get('verified', True) else 'NO'} |"
        )
    return header + "\n" + "\n".join(rows)


def _market_context_block(mc: dict[str, Any]) -> str:
    if not mc or (not mc.get("active") and not mc.get("pending")):
        return "_No active/pending context available._"
    def _rows(items):
        return "\n".join(
            f"- [{i.get('status')}] {i.get('id')} {i.get('neighborhood')} "
            f"{i.get('property_type')} {i.get('gla_sqft')} sqft - list {_fmt_money(i.get('list_price'))}"
            for i in items
        ) or "  - None"
    return (
        f"_{mc.get('note', '')}_\n\n"
        f"Active:\n{_rows(mc.get('active', []))}\n\n"
        f"Pending/Conditional:\n{_rows(mc.get('pending', []))}"
    )


def _adjustment_detail(comps: list[dict[str, Any]]) -> str:
    blocks = []
    for c in comps:
        adj = c.get("adjustments", [])
        if not adj:
            continue
        items = "\n".join(
            f"    - {a['factor']}: {a['detail']} -> {_fmt_money(a['amount'])}" for a in adj
        )
        blocks.append(
            f"- {c.get('id')} ({c.get('property_type')}, {c.get('gla_sqft')} sqft): "
            f"{_fmt_money(c.get('sale_price'))} -> {_fmt_money(c.get('adjusted_price'))} "
            f"(gross {c.get('gross_adjustment_pct', 0)*100:.1f}%)\n{items}"
        )
    return "\n".join(blocks) if blocks else "_No adjustments applied._"


def _risk_block(risk: dict[str, Any]) -> str:
    flags = risk.get("flags", [])
    if not flags:
        flag_txt = "- None"
    else:
        flag_txt = "\n".join(f"- [{f['severity'].upper()}] {f['message']}" for f in flags)
    m = risk.get("metrics", {})
    metrics_txt = (
        f"- Comps: {m.get('comp_count')}\n"
        f"- Dispersion (CoV): {m.get('coefficient_of_variation', 0)*100:.1f}%\n"
        f"- Mean gross adjustment: {m.get('mean_gross_adjustment_pct', 0)*100:.1f}%\n"
        f"- Median comp age: {m.get('median_comp_age_months')} mo\n"
        f"- Median comp distance: {m.get('median_comp_distance_km')} km\n"
        f"- Data quality score: {m.get('data_quality_score')}"
    )
    return f"Confidence: **{risk.get('confidence', 'n/a')}**\n\nFlags:\n{flag_txt}\n\nMetrics:\n{metrics_txt}"


def _assumptions_block(state: CompState, valued: bool) -> str:
    a = state.get("assignment", {})
    lines = [
        f"- Value is the *as-of* {a.get('effective_date', config.VALUATION_DATE.isoformat())} market value "
        "for the stated intended use only.",
        "- Comparable sales are adjusted to the effective date using the documented "
        f"market-appreciation rate ({config.MONTHLY_APPRECIATION*100:.1f}%/month).",
        "- All adjustment coefficients are from `src/config.py`; no figure in this "
        "report is produced by an LLM.",
        "- Legal/title and zoning findings are synthesized for demonstration and must "
        "be confirmed against authoritative land-titles and municipal sources.",
        "- Active/pending listing context is derived from recent nearby sales as a "
        "supply proxy, not a live MLS feed.",
    ]
    if not valued:
        lines.append("- A value could not be concluded; the file was returned for missing inputs.")
    return "\n".join(lines)


def report_writer_node(state: CompState) -> dict[str, Any]:
    s = state.get("subject", {})
    a = state.get("assignment", {})
    lt = state.get("legal_title", {})
    z = state.get("zoning", {})
    ms = state.get("market_scope", {})
    mc = state.get("market_context", {})
    valuation = state.get("valuation", {})
    risk = state.get("risk", {})
    comps = state.get("adjusted_comps", []) or state.get("ranked_comps", [])
    decision = state.get("human_decision", {})
    dq = state.get("data_quality", {})

    valued = bool(valuation) and valuation.get("method") != "no_comps"

    narrative = ""
    if valued:
        narrative = llm.write_narrative({"subject": s, "valuation": valuation, "risk": risk})
    else:
        narrative = (
            "Valuation could not be completed: the subject is missing critical fields "
            f"({', '.join(dq.get('missing_critical', []) or ['unknown'])}). "
            "Returned to the analyst for completion."
        )

    decision_txt = (
        f"- Action: **{decision.get('action', 'pending')}** by {decision.get('reviewer', 'n/a')}\n"
        f"- Note: {decision.get('note', '-')}"
    )
    if valuation.get("method") == "human_override":
        decision_txt += f"\n- Model estimate (pre-override): {_fmt_money(valuation.get('model_estimate'))}"

    if valued:
        val_block = (
            f"**Estimated value: {_fmt_money(valuation.get('point_estimate'))}**\n\n"
            f"- Range (low/mid/high): {_fmt_money(valuation.get('low'))} / "
            f"{_fmt_money(valuation.get('mid', valuation.get('point_estimate')))} / "
            f"{_fmt_money(valuation.get('high'))}\n"
            f"- Implied $/sqft: {_fmt_money(valuation.get('implied_ppsf'))}\n"
            f"- Method: {valuation.get('method')}\n"
            f"- Comps used: {valuation.get('comp_count')}"
        )
    else:
        val_block = "_Unable to produce a valuation (see rationale)._"

    md = f"""# KV Capital - Comparable Sales Valuation

_Generated {datetime.now():%Y-%m-%d %H:%M} | Effective date {a.get('effective_date', config.VALUATION_DATE.isoformat())}_

## Assignment
{_assignment_block(a)}

## Subject Property
{_subject_block(s)}

## Legal / Title
{_legal_block(lt)}

## Zoning & Highest-and-Best-Use
{_zoning_block(z)}

## Market Scope
{_scope_block(ms)}

## Reconciled Value
{val_block}

## Rationale
{narrative}

## Comparable Sales
{_comps_table(comps)}

## Active / Pending Market Context
{_market_context_block(mc)}

## Adjustment Detail
{_adjustment_detail(comps)}

## Risk Review
{_risk_block(risk)}

## Assumptions & Limiting Conditions
{_assumptions_block(state, valued)}

## Human Review / Sign-off
{decision_txt}

## Process Trace
""" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(state.get("trace", [])))

    report_json = {
        "assignment": a,
        "subject": s,
        "legal_title": lt,
        "zoning": z,
        "market_scope": ms,
        "market_context": mc,
        "verification": state.get("verification", {}),
        "valuation": valuation,
        "risk": risk,
        "comps": comps,
        "human_decision": decision,
        "data_quality": dq,
        "narrative": narrative,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    trace = state.get("trace", []) + ["report_writer: generated markdown + json workfile"]
    return {"report": {"markdown": md, "json": report_json, "narrative": narrative}, "trace": trace}
