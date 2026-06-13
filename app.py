"""Streamlit demo for the KV Capital comp-analysis agent.

Enter (or paste) a residential listing, run it through the LangGraph agent, and
review the ranked comps, itemized adjustments, valuation, and risk findings.
When a deal is Low confidence the graph pauses at the human-review interrupt and
this UI surfaces an Approve / Override / Reject control, then resumes the graph
on the same thread.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import uuid

import pandas as pd
import streamlit as st
from langgraph.types import Command

from src import config, llm
from src.data_store import dataset_summary
from src.graph import build_graph
from src.state import new_state

st.set_page_config(page_title="KV Capital - Comp Analysis Agent", layout="wide")

SAMPLES = {
    "Detached - Tuscany, Calgary (structured)": {
        "address": "123 Tuscany Ravine Rd NW", "neighborhood": "Tuscany", "city": "Calgary",
        "property_type": "Detached", "bedrooms": 4, "bathrooms": 3.0,
        "gla_sqft": 2200, "lot_size_sqft": 4800, "year_built": 2008,
    },
    "Luxury Detached - Mount Royal, Calgary (Low confidence -> human review)": {
        "address": "10 Mount Royal Cres SW", "neighborhood": "Mount Royal", "city": "Calgary",
        "property_type": "Detached", "bedrooms": 6, "bathrooms": 5.5,
        "gla_sqft": 4800, "lot_size_sqft": 9000, "year_built": 2018,
    },
    "Townhouse - Old Strathcona, Edmonton": {
        "address": "88 Whyte Ave NW", "neighborhood": "Old Strathcona", "city": "Edmonton",
        "property_type": "Townhouse", "bedrooms": 3, "bathrooms": 2.5,
        "gla_sqft": 1450, "lot_size_sqft": None, "year_built": 2012,
    },
}


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def _reset_run():
    for k in ("phase", "result", "payload", "thread", "final"):
        st.session_state.pop(k, None)


def _get_app():
    if "app" not in st.session_state:
        st.session_state.app = build_graph()
    return st.session_state.app


def _is_paused(result: dict) -> bool:
    return bool(result.get("__interrupt__"))


def _interrupt_payload(result: dict) -> dict:
    intr = result.get("__interrupt__")
    if not intr:
        return {}
    first = intr[0]
    return getattr(first, "value", first) or {}


_ROOT = os.path.dirname(os.path.abspath(__file__))


def _discover_cases() -> list[str]:
    """Folder names in the project root that look like a valuation case.

    A case folder is recognized by the presence of the ``00_assignment``
    section directory from ``config.CASE_LAYOUT``.
    """
    assignment_dir = config.CASE_LAYOUT["assignment"]["dir"]
    out: list[str] = []
    for name in sorted(os.listdir(_ROOT)):
        full = os.path.join(_ROOT, name)
        if os.path.isdir(full) and os.path.isdir(os.path.join(full, assignment_dir)):
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_workflow_context(state: dict):
    """Assignment, legal/title, zoning, and market-scope findings."""
    a = state.get("assignment", {})
    lt = state.get("legal_title", {})
    z = state.get("zoning", {})
    ms = state.get("market_scope", {})
    mc = state.get("market_context", {})

    with st.expander("Assignment, legal/title, zoning & market scope"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Assignment**")
            st.write({
                "Client": a.get("client"), "Borrower": a.get("borrower"),
                "Intended use": a.get("intended_use"),
                "Effective date": a.get("effective_date"),
            })
            st.markdown("**Legal / Title**")
            st.write({
                "Address confirmed": lt.get("address_confirmed"),
                "Legal description": lt.get("legal_description"),
                "Parcel ID": lt.get("parcel_id"),
                "Title status": lt.get("title_status"),
            })
        with c2:
            st.markdown("**Zoning & HBU**")
            st.write({
                "Zoning": z.get("zoning_code"),
                "Permitted use": z.get("permitted_use"),
                "Conforming": z.get("conforming"),
                "Highest & best use": z.get("highest_and_best_use"),
            })
            st.markdown("**Market scope**")
            st.write({
                "Radius (km)": ms.get("radius_km"),
                "Recency (months)": ms.get("recency_months"),
                "GLA band": ms.get("gla_band"),
                "Comparable types": ms.get("property_types"),
            })

    if mc and (mc.get("active") or mc.get("pending")):
        with st.expander("Active / pending market context"):
            rows = (mc.get("active", []) + mc.get("pending", []))
            st.dataframe(
                pd.DataFrame([{
                    "Status": r.get("status"), "Comp": r.get("id"),
                    "Neighborhood": r.get("neighborhood"), "Type": r.get("property_type"),
                    "GLA": r.get("gla_sqft"), "List price": r.get("list_price"),
                } for r in rows]),
                column_config={"List price": st.column_config.NumberColumn(format="$%d")},
                hide_index=True, use_container_width=True,
            )
            st.caption(mc.get("note", ""))


def render_results(state: dict):
    s = state.get("subject", {})
    v = state.get("valuation", {})
    risk = state.get("risk", {})
    comps = state.get("adjusted_comps", []) or state.get("ranked_comps", [])
    decision = state.get("human_decision", {})

    valued = bool(v) and v.get("method") != "no_comps"

    st.subheader("Reconciled Value")
    if valued:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estimated value", f"${v.get('point_estimate', 0):,.0f}")
        c2.metric("Range", f"${v.get('low', 0):,.0f} - ${v.get('high', 0):,.0f}")
        c3.metric("Implied $/sqft", f"${v.get('implied_ppsf', 0):,.0f}")
        conf = risk.get("confidence", "n/a")
        c4.metric("Confidence", conf)
        if v.get("method") == "human_override":
            st.info(f"Reviewer override applied. Model estimate was "
                    f"${v.get('model_estimate', 0):,.0f}.")
    else:
        st.warning("Unable to produce a valuation - subject is missing critical fields: "
                   f"{', '.join(state.get('data_quality', {}).get('missing_critical', []))}")

    if decision:
        st.caption(f"Human sign-off: **{decision.get('action', 'n/a')}** "
                   f"by {decision.get('reviewer', 'n/a')} - {decision.get('note', '')}")

    render_workflow_context(state)

    # Rationale
    report = state.get("report", {})
    if report.get("narrative"):
        st.subheader("Rationale")
        st.write(report["narrative"])

    # Risk
    st.subheader("Risk Review")
    flags = risk.get("flags", [])
    if flags:
        for f in flags:
            (st.error if f["severity"] == "high" else st.warning)(
                f"[{f['severity'].upper()}] {f['message']}")
    else:
        st.success("No material risk flags.")
    if risk.get("metrics"):
        st.json(risk["metrics"], expanded=False)

    # Comps
    if comps:
        st.subheader(f"Comparable Sales (top {len(comps)})")
        df = pd.DataFrame([{
            "Comp": c.get("id"), "Neighborhood": c.get("neighborhood"),
            "Type": c.get("property_type"), "Beds": c.get("bedrooms"),
            "Baths": c.get("bathrooms"), "GLA": c.get("gla_sqft"),
            "Sold": str(c.get("sale_date"))[:10], "Dist (km)": round(c.get("distance_km", 0), 2),
            "Age (mo)": round(c.get("months_ago", 0), 1),
            "Sale": c.get("sale_price"), "Net Adj": c.get("net_adjustment"),
            "Adjusted": c.get("adjusted_price"), "Similarity": c.get("similarity"),
            "Weight": c.get("weight"),
        } for c in comps])
        st.dataframe(
            df,
            column_config={
                "Sale": st.column_config.NumberColumn(format="$%d"),
                "Net Adj": st.column_config.NumberColumn(format="$%d"),
                "Adjusted": st.column_config.NumberColumn(format="$%d"),
            },
            hide_index=True, use_container_width=True,
        )

        st.subheader("Adjustment Detail")
        for c in comps:
            with st.expander(
                f"{c.get('id')} - {c.get('property_type')} {c.get('gla_sqft')} sqft | "
                f"${c.get('sale_price', 0):,.0f} -> ${c.get('adjusted_price', 0):,.0f} "
                f"(gross {c.get('gross_adjustment_pct', 0) * 100:.1f}%)"
            ):
                adf = pd.DataFrame(c.get("adjustments", []))
                if not adf.empty:
                    st.dataframe(
                        adf, hide_index=True, use_container_width=True,
                        column_config={"amount": st.column_config.NumberColumn(format="$%d")},
                    )

    # Report download + trace
    if report.get("markdown"):
        st.subheader("Underwriting Report")
        with st.expander("Full markdown report"):
            st.markdown(report["markdown"])
        st.download_button("Download report (.md)", report["markdown"],
                           file_name="kv_valuation_report.md")
    with st.expander("Process trace (node-by-node)"):
        for i, t in enumerate(state.get("trace", []), 1):
            st.text(f"{i}. {t}")


def render_human_review(payload: dict):
    st.subheader("Human review required")
    st.warning(f"Reason: {payload.get('reason', 'review required')} "
               f"(confidence: {payload.get('confidence', 'n/a')})")
    est = payload.get("model_estimate")
    if est:
        rng = payload.get("range", [None, None])
        st.write(f"Model estimate: **${est:,.0f}** "
                 f"(range ${rng[0]:,.0f} - ${rng[1]:,.0f})")
    for f in payload.get("flags", []):
        st.caption(f"- [{f['severity'].upper()}] {f['message']}")

    with st.form("human_review_form"):
        action = st.radio("Decision", ["approve", "override", "reject"], horizontal=True)
        override_value = st.number_input(
            "Override value ($) - used only if action is 'override'",
            min_value=0.0, value=float(est or 0), step=10000.0,
        )
        reviewer = st.text_input("Reviewer", value="loan_officer")
        note = st.text_input("Note", value="")
        submitted = st.form_submit_button("Submit decision")

    if submitted:
        decision = {"action": action, "reviewer": reviewer, "note": note}
        if action == "override":
            decision["override_value"] = override_value
        app = _get_app()
        cfg = {"configurable": {"thread_id": st.session_state.thread}}
        result = app.invoke(Command(resume=decision), config=cfg)
        if _is_paused(result):
            st.session_state.result = result
            st.session_state.payload = _interrupt_payload(result)
            st.session_state.phase = "paused"
        else:
            st.session_state.final = app.get_state(cfg).values
            st.session_state.phase = "done"
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("KV Capital")
    st.caption("Comparable-sales valuation agent (LangGraph)")
    summary = dataset_summary()
    st.metric("Comps in dataset", f"{summary['records']:,}")
    st.caption(f"Cities: {summary['cities']}")
    st.caption(f"Median sale: ${summary['median_price']:,}")
    st.caption(f"Valuation date: {config.VALUATION_DATE:%Y-%m-%d}")
    st.divider()
    if llm.llm_available():
        st.success("LLM: OpenAI enabled")
    else:
        st.error("LLM: no API key — set OPENAI_API_KEY to run (the agent nodes "
                 "require it).")
    st.caption("The specialist nodes are tool-calling LLM agents; an OPENAI_API_KEY "
               "is required. All valuation math is deterministic.")
    if os.getenv("KV_FAST_PATH", "auto").lower() not in ("0", "false", "no", "off"):
        st.caption("**Case folder runs:** fast path is on by default "
                   "(`KV_FAST_PATH=auto`) — tools run directly, LLM only for narratives. "
                   "Set `KV_FAST_NARRATIVE=1` in `.env` to skip those too.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("Comparable-Sales Valuation Agent")
st.caption("Given a subject property, retrieve and rank comparable sales, compute "
           "appraisal-style adjustments, and produce an explained valuation.")

phase = st.session_state.get("phase", "input")

if phase in ("input",) or "phase" not in st.session_state:
    mode = st.radio(
        "Input mode",
        ["Sample property", "Structured form", "Free-text listing", "Case folder"],
        horizontal=True,
    )
    raw_input = None
    case_dir = None

    if mode == "Sample property":
        choice = st.selectbox("Sample subject", list(SAMPLES.keys()))
        raw_input = {k: v for k, v in SAMPLES[choice].items() if v is not None}
        st.json(raw_input, expanded=False)

    elif mode == "Structured form":
        col1, col2, col3 = st.columns(3)
        with col1:
            address = st.text_input("Address", "500 Riverside Dr SW")
            neighborhood = st.selectbox("Neighborhood", list(config.NEIGHBORHOODS.keys()))
            ptype = st.selectbox("Property type", config.PROPERTY_TYPES)
        with col2:
            beds = st.number_input("Bedrooms", 0, 10, 3)
            baths = st.number_input("Bathrooms", 0.0, 12.0, 2.5, step=0.5)
            gla = st.number_input("GLA (sqft)", 300, 12000, 1800, step=50)
        with col3:
            lot = st.number_input("Lot size (sqft, 0 if n/a)", 0, 60000, 0, step=100)
            year = st.number_input("Year built", 1890, config.VALUATION_DATE.year, 2010)
        city = config.NEIGHBORHOODS[neighborhood]["city"]
        raw_input = {
            "address": address, "neighborhood": neighborhood, "city": city,
            "property_type": ptype, "bedrooms": int(beds), "bathrooms": float(baths),
            "gla_sqft": int(gla), "year_built": int(year),
        }
        if lot > 0:
            raw_input["lot_size_sqft"] = int(lot)

    elif mode == "Free-text listing":
        text = st.text_area(
            "Paste a listing", height=120,
            value="Charming 3 bed, 2.5 bath townhouse in Bridgeland, Calgary. "
                  "About 1,500 sqft, built in 2014.")
        raw_input = text

    else:  # Case folder
        st.caption("Run the full workflow over a structured case folder "
                   "(`valuation_case_XXX/`). Build a real Calgary case with "
                   "`python data/download_case.py`.")
        cases = _discover_cases()
        options = cases + ["Enter a path manually…"]
        choice = st.selectbox("Case folder", options) if cases else "Enter a path manually…"
        if not cases:
            st.info("No case folders found in the project root. Create one with "
                    "`python data/download_case.py`, or enter a path below.")
        if choice == "Enter a path manually…":
            case_dir = st.text_input("Case folder path", value="").strip() or None
        else:
            case_dir = choice
        if case_dir:
            st.code(case_dir, language="text")

    run_disabled = (mode == "Case folder" and not case_dir)
    if st.button("Run valuation", type="primary", disabled=run_disabled):
        if not llm.llm_available():
            st.error("OPENAI_API_KEY is not set. The agent nodes are tool-calling "
                     "LLM agents and require an API key to run.")
            st.stop()
        thread = str(uuid.uuid4())
        st.session_state.thread = thread
        app = _get_app()
        cfg = {"configurable": {"thread_id": thread}}
        result = app.invoke(new_state(raw_input, case_dir=case_dir), config=cfg)
        if _is_paused(result):
            st.session_state.result = result
            st.session_state.payload = _interrupt_payload(result)
            st.session_state.phase = "paused"
        else:
            st.session_state.final = app.get_state(cfg).values
            st.session_state.phase = "done"
        st.rerun()

elif phase == "paused":
    render_human_review(st.session_state.payload)
    st.divider()
    if st.button("Start over"):
        _reset_run()
        st.rerun()

elif phase == "done":
    render_results(st.session_state.final)
    st.divider()
    if st.button("Run another property", type="primary"):
        _reset_run()
        st.rerun()
