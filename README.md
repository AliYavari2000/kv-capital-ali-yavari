# KV Capital — Comparable-Sales Valuation Agent

An AI agent that, given a subject residential property, **retrieves and ranks
comparable recent sales, computes appraisal-style adjustments, produces a
valuation estimate with a confidence range, flags risk, and explains its
reasoning** — with a human-in-the-loop checkpoint for low-confidence deals.

It is built as a [LangGraph](https://langchain-ai.github.io/langgraph/) state
machine that mirrors a standard residential appraisal workflow (11 specialist
agent nodes plus a human sign-off), runs on a synthesized Alberta sales dataset,
and ships with a headless CLI and a Streamlit demo UI.

This automates the **comp-analysis bottleneck** in KV Capital's residential
lending workflow: finding comparable sales and using them to estimate value.

---

## Table of contents

- [What it does](#what-it-does)
- [Why this design](#why-this-design)
- [Quickstart](#quickstart)
- [Using the agent](#using-the-agent)
  - [Streamlit UI](#1-streamlit-ui-apppy)
  - [Real Calgary case download](#2-real-calgary-case-download-datadownload_casepy)
- [Architecture](#architecture)
- [The agent nodes](#the-agent-nodes)
- [Valuation methodology](#valuation-methodology)
- [Case folders](#case-folders)
- [Calgary / Alberta data sources](#calgary--alberta-data-sources)
- [Configuration](#configuration)
- [Repository layout](#repository-layout)
- [Scoping notes](#scoping-notes-whats-in-whats-deliberately-out)

---

## What it does

Given a subject property (as a structured record, a free-text listing, or a
structured **case folder**), the agent:

1. **Frames the assignment** — client, borrower, intended use, effective date.
2. **Normalizes the subject** — type, beds/baths, GLA, lot, age, condition; runs
   a data-quality gate that routes incomplete subjects to a human.
3. **Establishes context** — legal/title, zoning and highest-and-best-use, and a
   market scope (search radius, recency window, GLA band, comparable types).
4. **Retrieves and ranks comparable sales** with progressive widening if too few
   candidates are found, and attaches active/pending listings for context.
5. **Verifies and normalizes** comps (cross-checks price/date/GLA/lot; standardizes
   units, condition, basement/garage).
6. **Adjusts** each comp toward the subject with market-supported, deterministic
   adjustments (time, type, GLA, beds/baths, lot, age).
7. **Reconciles** into a weighted point estimate plus a low/mid/high range and a
   High/Medium/Low confidence.
8. **Scores risk** and, when confidence is **Low**, pauses for a human reviewer
   to approve / override / reject.
9. **Writes** an underwriting memo (Markdown + JSON) with the comp grid, itemized
   adjustments, assumptions, citations, limitations, and a grounded narrative.

---

## Why this design

Underwriting needs numbers a credit committee can defend, so the architecture
draws a hard line:

- **The valuation math is deterministic and auditable.** Retrieval, similarity
  ranking, the sales-comparison adjustment grid, and the final reconciliation
  are pure functions with documented coefficients in
  [`src/config.py`](src/config.py). Every dollar in the report traces back to a
  formula. **The LLM never invents a number.**
- **The LLM does the language work only**: parsing messy free-text listings into
  structured fields, classifying/normalizing the subject and comps, and writing
  the narrative rationale from already-computed facts. The specialist nodes are
  tool-calling LLM agents, so **an `OPENAI_API_KEY` is required** to run the
  graph — there is no offline fallback for the agent nodes (see
  [`src/llm.py`](src/llm.py), `run_tool_agent`).
- **Synthetic data, transparent model.** Sales are generated from the same
  contributory-value model the agent inverts (see
  [`data/generate_data.py`](data/generate_data.py)), so the recovered valuation
  is checkable against ground truth. On the clean Tuscany sample the agent
  estimates **~$930k** against a modeled value of **~$930k**.

---

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Configure the LLM — REQUIRED. The specialist nodes are tool-calling agents.
cp .env.example .env     # then set OPENAI_API_KEY (and optionally KV_LLM_MODEL)

# 2) Generate the synthetic comps dataset (writes data/comps.csv) — required once.
python data/generate_data.py

# 3) Launch the Streamlit UI (the only entry point)
streamlit run app.py
```

> **`OPENAI_API_KEY` is required.** The agent nodes call the model through a
> tool-calling loop (`src/llm.py:run_tool_agent`) and raise
> `LLMUnavailableError` if no key is set. `KV_LLM_MODEL` is optional (defaults
> to `gpt-4o-mini`).

---

## Using the agent

The Streamlit app is the single entry point for running the agent.

### 1) Streamlit UI (`app.py`)

```bash
streamlit run app.py
```

Choose one of four **input modes**:

| Mode | What it does |
|------|--------------|
| **Sample property** | Run a built-in sample subject. |
| **Structured form** | Type the subject fields (address, neighborhood, type, beds/baths, GLA, lot, year). |
| **Free-text listing** | Paste a listing; the LLM parses it into structured fields. |
| **Case folder** | Run the full workflow over a structured `valuation_case_XXX/` folder — including the real Calgary case from `download_case.py` or **any other case folder** you provide. |

In **Case folder** mode the app auto-discovers any `valuation_case_*` folders in
the project root (recognized by their `00_assignment/` section) and lists them
in a dropdown; you can also choose **"Enter a path manually…"** to point at a
case folder anywhere on disk (absolute path, or relative to the project root).

The UI shows the ranked comps, itemized adjustments, valuation, range, and risk
findings. When a deal is **Low** confidence the graph pauses at the
human-review interrupt and the UI surfaces an **Approve / Override / Reject**
control, then resumes the graph on the same thread.

### 2) Real Calgary case download (`data/download_case.py`)

Assembles a real Calgary subject into the standard case-folder layout using
authoritative **Open Calgary** public datasets (property assessment, building
permits, parcel geometry). Comparable *sale* data is not publicly available
(MLS/Pillar 9 is licensed), so the sold/active comp CSVs are filled from the
project's synthetic `data/comps.csv` filtered to the subject's neighbourhood.

```bash
# Default: Signal Hill detached -> writes valuation_case_001/
python data/download_case.py

# Pick a community and auto-select a representative subject
python data/download_case.py --community "TUSCANY" --address ""

# Target a specific address (must match the chosen community)
python data/download_case.py --community "SIGNAL HILL" --address "2952 SIGNAL HILL DR SW"

# Custom output directory
python data/download_case.py --out valuation_case_custom --community "EVANSTON" --address ""
```

> **Note:** `--address` must match a parcel inside `--community`, or the script
> errors with "Address not found". Pass `--address ""` to let it auto-pick a
> representative detached parcel for the community. Requires network access and
> a prior `python data/generate_data.py` (for the comps substitute).

Then run the full workflow on the downloaded case from the Streamlit UI: launch
`streamlit run app.py`, pick **Case folder** mode, and select
`valuation_case_001` (or whatever `--out` you chose) from the dropdown.

---

## Architecture

```
assignment_intake -> subject_property
    subject_property --(critical field missing)--> human_review
    subject_property --(ok)--> legal_title -> zoning_hbu -> market_scope
        -> comp_retrieval -> fact_verification -> normalization
        -> adjustment_engine -> reconciliation
            reconciliation --(Low confidence)--> human_review
            reconciliation --(ok)--> report_writer
    human_review --(reject + valued)--> market_scope   (single re-pull, then proceeds)
    human_review --(approve/override/else)--> report_writer
report_writer -> END
```

The graph is wired in [`src/graph.py`](src/graph.py) with a `MemorySaver`
checkpointer so the human-review `interrupt` can pause and resume on a thread.
The Streamlit app drives this interrupt/resume loop directly; `run_headless()`
in `src/graph.py` is a convenience helper that auto-approves the review for
scripted, non-interactive runs.

---

## The agent nodes

| # | Agent / node | File | Responsibility |
|---|------|------|----------------|
| 1 | `AssignmentIntakeNode` | [`src/nodes/node1_assignment_intake.py`](src/nodes/node1_assignment_intake.py) | Define assignment + effective date; parse borrower/listing info, property type, intended use (LLM agent) |
| 2 | `SubjectPropertyNode` | [`src/nodes/node2_subject_property.py`](src/nodes/node2_subject_property.py) | Inspect/measure subject: beds, baths, GLA, lot, age, condition; attach lat/lon; data-quality gate |
| 3 | `LegalTitleNode` | [`src/nodes/node3_legal_title.py`](src/nodes/node3_legal_title.py) | Confirm address, legal description, parcel/LINC ID, ownership/title issues (folds in assessment/tax) |
| 4 | `ZoningHBUNode` | [`src/nodes/node4_zoning_hbu.py`](src/nodes/node4_zoning_hbu.py) | Zoning, permitted use, nonconforming issues, highest-and-best-use (folds in permits/survey) |
| 5 | `MarketScopeNode` | [`src/nodes/node5_market_scope.py`](src/nodes/node5_market_scope.py) | Choose search radius / time window / property-type filters |
| 6 | `CompRetrievalNode` | [`src/nodes/node6_comp_retrieval.py`](src/nodes/node6_comp_retrieval.py) | Pull sold comps (with widening), rank by similarity, attach active/pending context |
| 7 | `FactVerificationNode` | [`src/nodes/node7_fact_verification.py`](src/nodes/node7_fact_verification.py) | Cross-check price, date, GLA, lot size against plausibility rules |
| 8 | `NormalizationNode` | [`src/nodes/node8_normalization.py`](src/nodes/node8_normalization.py) | Standardize units, basement/garage/condition, GLA, lot across subject + comps |
| 9 | `AdjustmentEngineNode` | [`src/nodes/node9_adjustment_engine.py`](src/nodes/node9_adjustment_engine.py) | Market-supported adjustments (time, type, size, beds/baths, lot, age) -> adjusted price |
| 10 | `ReconciliationNode` | [`src/nodes/node10_reconciliation.py`](src/nodes/node10_reconciliation.py) | Weighted reconciliation -> low/mid/high + confidence (folds in legal/zoning/verification flags) |
| 11 | `ReportWriterNode` | [`src/nodes/node11_report_writer.py`](src/nodes/node11_report_writer.py) | Markdown + JSON memo: comp grid, assumptions, citations, limitations, grounded narrative |
| 12 | Human Reviewer / Sign-off | [`src/nodes/human_review.py`](src/nodes/human_review.py) | `interrupt` for approve / override / reject when confidence is Low |

---

## Valuation methodology

Classic **sales-comparison approach**. Each comp's sale price is adjusted toward
the subject (in valuation-date dollars):

- **Time / market** — appreciate the sale to the valuation date
  (`MONTHLY_APPRECIATION`, ~0.4%/mo).
- **Property type** — convert the comp's pricing tier to the subject's
  (`TYPE_PPSF_MULTIPLIER`).
- **GLA** — size delta at neighborhood/type implied `$/sqft`.
- **Bedrooms / Bathrooms / Lot size / Effective age** — additive contributory
  values (`BED_VALUE`, `BATH_VALUE`, `LOT_PPSF`, `AGE_VALUE_PER_YEAR`).

The estimate is a weighted reconciliation of the adjusted comps, where weight
favors the most similar and least-adjusted comps
(`weight = similarity^2 / (1 + 4 * gross_adjustment_pct)`). The range comes from
the weighted dispersion of adjusted values. Confidence (High/Medium/Low) is
derived from risk flags (comp count, dispersion, gross-adjustment level, comp
age/distance, data quality); **Low** routes the deal to human review.

---

## Case folders

A **case** is a single appraisal assignment delivered as a structured folder.
The folder→node mapping lives in `config.CASE_LAYOUT`; the read-only ingestion
view is [`src/case_store.py`](src/case_store.py). Structured files
(`.csv` / `.json` / `.geojson` / `.md`) are parsed for real; heavy formats
(`.pdf` / `.jpg` / `.tif` / `.docx` / `.xlsx`) are exposed as typed *document
hooks* so per-node tools (PDF text extraction, OCR, etc.) can slot in later.

```
valuation_case_XXX/
  00_assignment/            effective_date.json + lender/borrower docs
  01_subject_property/      subject_listing.csv + photos, floor plan, RMS report
  02_legal_title/           title, RPR, compliance, encumbrances
  03_assessment_tax/        assessment_open_data.csv + assessment/tax PDFs
  04_zoning_land_use/       *.geojson (parcel/community/land-use), zoning notes
  05_permits_rpr_surveys/   building/development permits, survey measurements
  06_comparables/           sold/active/pending/rejected/verified comp CSVs + sheets
  07_market_context/        price index, absorption, days-on-market, CREB reports
  08_workflow_outputs/      (reserved for run artifacts)
  09_final_package/         (reserved for final deliverable)
  data_source_manifest.json case-file -> authoritative-source mapping
```

Create one with `data/download_case.py` (real Open Calgary subject), then run it
from the Streamlit UI's **Case folder** input mode.

---

## Calgary / Alberta data sources

The workflow is modeled on real Calgary residential appraisal practice. Each
case-folder file maps to an authoritative Alberta source (see
[`src/data_sources.py`](src/data_sources.py) and `data_source_manifest.json` in
the case root):

| Data need | Source | Case folder |
|-----------|--------|-------------|
| Appraisal standards | AIC / CUSPAP | referenced in assumptions |
| Residential measurement | RECA RMS | `01_subject_property/rms_measurement_report.pdf` |
| Assessment / property details | Calgary Assessment + Open Calgary | `03_assessment_tax/` |
| Parcel / address / zoning GIS | Open Calgary | `04_zoning_land_use/*.geojson` |
| Building & development permits | Open Calgary | `05_permits_rpr_surveys/` |
| Land title & survey plans | Alberta SPIN2 | `02_legal_title/` |
| MLS sold/active listings | Pillar 9 / MLS | `06_comparables/` |
| Market context | CREB | `07_market_context/` |

`data/download_case.py` pulls live from Open Calgary's
[Current Year Property Assessments](https://data.calgary.ca/resource/4bsw-nn7w.json)
(`4bsw-nn7w`) and [Building Permits](https://data.calgary.ca/resource/c2es-76ed.json)
(`c2es-76ed`). MLS/title/CREB exports are licensed, so those slots use synthetic
or placeholder content (clearly marked in the manifest provenance).

---

## Configuration

All model knobs live in [`src/config.py`](src/config.py):

- `VALUATION_DATE`, `MONTHLY_APPRECIATION` — market anchor and time trend.
- `NEIGHBORHOODS` — centroids + base `$/sqft` for 19 Calgary/Edmonton areas.
- `PROPERTY_TYPES`, `TYPE_PPSF_MULTIPLIER`, `TYPE_HAS_LOT` — type pricing.
- `BED_VALUE`, `BATH_VALUE`, `LOT_PPSF`, `AGE_VALUE_PER_YEAR` — adjustment coefficients.
- `RANKING_WEIGHTS`, `TOP_N_COMPS` — similarity ranking and comp count.
- `RETRIEVAL` — search radius/recency/GLA band and progressive-widening policy.
- `CRITICAL_FIELDS`, `SOFT_FIELDS` — the data-quality gate.
- `RISK`, `HUMAN_REVIEW_ON` — risk thresholds and the confidence levels that force review.
- `CASE_LAYOUT`, `COLUMN_ALIASES` — case-folder layout and CSV header normalization.

Environment (`.env`): `OPENAI_API_KEY` (**required** — the agent nodes are
tool-calling LLM agents with no offline fallback), `KV_LLM_MODEL` (optional,
default `gpt-4o-mini`).

### Performance / latency

A full run was ~60–90s because **nine sequential LLM agent nodes** each make
many tool-calling round trips (~50+ API calls), plus two narrative calls at the
end. The valuation math itself is instant.

| Setting | Effect |
|---------|--------|
| **`KV_FAST_PATH=auto`** (default) | For **case folders**, runs each node's tools in a fixed script — **no LLM orchestration**. Cuts most API calls; narratives still use the LLM. |
| **`KV_FAST_PATH=0`** | Full LLM agent loop on every node (slowest; use for free-text / exploratory runs). |
| **`KV_FAST_NARRATIVE=1`** | Templated reconciliation + report prose instead of LLM (~2 fewer API calls). |

Recommended for Calgary case runs in Streamlit (**Case folder** mode):

```bash
# .env — fast case-folder runs (default auto path + optional template narratives)
OPENAI_API_KEY=sk-...
KV_FAST_NARRATIVE=1   # optional: ~5–15s end-to-end instead of ~60–90s
```

Free-text and structured-form inputs still use the LLM agent loop unless you set
`KV_FAST_PATH=1` globally (not recommended for messy listings).

---

## Repository layout

```
data/generate_data.py     # synthetic Alberta sales generator (transparent model)
data/download_case.py     # build a real Calgary case folder from Open Calgary open data
data/comps.csv            # generated dataset (~2,500 records)
src/config.py             # market model, coefficients, weights, thresholds, case layout
src/state.py              # shared LangGraph state (TypedDict)
src/data_store.py         # load + filtered retrieval with widening
src/case_store.py         # case-folder ingestion (CSV/JSON/GeoJSON/MD + document hooks)
src/data_sources.py       # Alberta authoritative-source mapping + manifest
src/valuation_math.py     # haversine, similarity, adjustment grid
src/fast_path.py          # deterministic tool scripts (skip LLM orchestration)
src/llm.py                # OpenAI tool-calling agent loop + parsing/narrative helpers
src/nodes/*.py            # the agent nodes (node1..node11 + human_review)
src/graph.py              # StateGraph wiring + checkpointer
app.py                    # Streamlit UI — sole entry point (samples / form / free-text / case folder)
```

---

## Scoping notes (what's in, what's deliberately out)

**In scope (focused + shippable):** residential subjects; synthesized but
realistic Calgary/Edmonton sales; deterministic, explainable valuation; risk
scoring; human-in-the-loop; a usable UI and CLI; real Open Calgary case assembly.

**Out of scope (clear extension points):**

- **Commercial borrowers** — the same graph generalizes: swap the adjustment
  grid for income/cap-rate logic and add property-class filters in retrieval.
  The intake/normalize/quality/risk/human/report scaffolding is reused as-is.
- **Live MLS data** — `src/data_store.py` is the single seam to replace the CSV
  with an MLS / land-titles feed; no other node changes.
- **Persistence/auth** — the checkpointer is in-memory (`MemorySaver`); swap for
  a durable saver (e.g. Postgres) for multi-session review queues.
- The `reject` path performs a single comp re-pull then proceeds with the model
  estimate (guarded against loops); in production an analyst would amend inputs
  before re-pulling.

All sales data is **synthetic** — no real listings or PII. Municipal facts
pulled by `download_case.py` are real public Open Calgary records.
