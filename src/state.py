"""Shared graph state for the comp-analysis agent.

A single ``TypedDict`` flows through every node. Nodes return partial dicts that
LangGraph merges into the state. Everything needed to audit a valuation -- the
assignment definition, the inspected subject, legal/title and zoning findings,
the market scope, the retrieved + verified + normalized comps with itemized
adjustments, the reconciled value, the human sign-off, and a running trace --
lives here.

The pipeline follows a standard residential appraisal order:

    assignment_intake -> subject_property -> legal_title -> zoning_hbu
        -> market_scope -> comp_retrieval -> fact_verification
        -> normalization -> adjustment_engine -> reconciliation
        -> report_writer -> human_review (final sign-off)
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages

Confidence = Literal["High", "Medium", "Low"]


class AdjustmentLine(TypedDict):
    """One itemized sales-comparison adjustment applied to a comp."""
    factor: str            # e.g. "GLA", "Bedrooms", "Time/Market"
    detail: str            # human-readable explanation of the delta
    amount: float          # signed dollars added to the comp's sale price


class RankedComp(TypedDict, total=False):
    """A candidate comparable with similarity, adjustments, and adjusted value."""
    id: str
    address: str
    neighborhood: str
    city: str
    property_type: str
    bedrooms: int
    bathrooms: float
    gla_sqft: int
    lot_size_sqft: int
    year_built: int
    sale_date: str
    sale_price: float
    distance_km: float
    months_ago: float
    similarity: float                 # 0..1 overall similarity to subject
    similarity_breakdown: dict[str, float]
    adjustments: list[AdjustmentLine]
    gross_adjustment: float           # sum of |adjustment|
    net_adjustment: float             # sum of adjustments (signed)
    adjusted_price: float             # sale_price + net_adjustment
    weight: float                     # reconciliation weight in valuation


class Valuation(TypedDict, total=False):
    point_estimate: float
    low: float
    high: float
    implied_ppsf: float
    method: str
    comp_count: int


class RiskReview(TypedDict, total=False):
    confidence: Confidence
    flags: list[dict[str, str]]       # {"code", "severity", "message"}
    requires_human_review: bool
    metrics: dict[str, float]


class HumanDecision(TypedDict, total=False):
    action: Literal["approve", "override", "reject"]
    override_value: Optional[float]
    note: str
    reviewer: str


class Assignment(TypedDict, total=False):
    """Scope of the appraisal assignment (who/why/as-of-when)."""
    client: str
    borrower: str
    intended_use: str
    effective_date: str               # valuation "as of" date (ISO)
    report_date: str                  # date the assignment was taken
    property_type: str
    # Filled by the Node 1 LLM agent:
    assignment_type: str              # purchase | refinance | construction_loan | ...
    valuation_purpose: str
    special_requirements: list[str]   # lender/client special instructions
    required_documents: dict[str, Any]  # {required, present, missing}
    missing_info: list[str]           # critical fields still missing
    workfile_id: str
    requires_human_review: bool       # intake-level escalation
    escalations: list[dict[str, Any]]


class LegalTitle(TypedDict, total=False):
    """Confirmation of the subject's legal identity and title status."""
    address_confirmed: bool
    legal_description: str
    parcel_id: str
    title_status: str                 # e.g. "Clear", "Encumbered", "Unverified"
    registered_owner: str
    flags: list[dict[str, str]]       # {"code", "severity", "message"}


class ZoningHBU(TypedDict, total=False):
    """Zoning / land-use review and highest-and-best-use conclusion."""
    zoning_code: str
    permitted_use: str
    conforming: bool                  # is current use conforming?
    highest_and_best_use: str
    flags: list[dict[str, str]]


class MarketScope(TypedDict, total=False):
    """The comp-search envelope chosen for this subject."""
    radius_km: float
    recency_months: int
    gla_band: float                   # +/- fraction of subject GLA
    property_types: list[str]         # allowed comp property types
    rationale: str


class CompState(TypedDict, total=False):
    # Inputs
    raw_input: Any                    # dict of fields OR free-text listing str
    intake_source: str                # "structured" | "free_text" | "case_folder"
    case_dir: str                     # path to a valuation_case_XXX/ folder (optional)
    documents: dict[str, Any]         # per-section parsed data + typed document hooks
    data_sources: dict[str, Any]      # Calgary/Alberta source registry + case file mapping
    assignment: Assignment            # assignment definition + effective date
    subject: dict[str, Any]           # inspected + normalized subject property

    # Legal / zoning
    legal_title: LegalTitle
    zoning: ZoningHBU

    # Data quality
    data_quality: dict[str, Any]      # score, missing fields, issues

    # Market scope + retrieval
    market_scope: MarketScope         # radius / recency / type filters used
    candidates: list[dict[str, Any]]  # candidate comps from the store
    retrieval_meta: dict[str, Any]    # filters used, widening attempts
    market_context: dict[str, Any]    # active + pending/conditional listings
    ranked_comps: list[RankedComp]    # top-N selected + scored
    rejected_comps: list[dict[str, Any]]  # rejected comp candidates + reasons

    # Verification + normalization
    verification: dict[str, Any]      # cross-check results + flags per comp
    normalization: dict[str, Any]     # assumptions + flagged outliers

    # Adjustment + valuation
    adjusted_comps: list[RankedComp]  # ranked comps with adjustments filled in
    adjustment_grid: list[dict[str, Any]]
    adjustment_flags: list[dict[str, str]]
    sensitivity: dict[str, Any]
    valuation: Valuation

    # Risk + human review
    risk: RiskReview
    human_decision: HumanDecision
    rerun_count: int

    # Output
    report: dict[str, Any]            # {"markdown", "json"}

    # Bookkeeping
    workfile_id: str                  # unique valuation/workfile ID (Node 1)
    evidence: list[dict[str, Any]]    # evidence trail (source docs/facts logged by agents)
    trace: list[str]                  # ordered node-by-node log
    messages: Annotated[list, add_messages]
    errors: list[str]


def new_state(raw_input: Any = None, *, case_dir: Optional[str] = None) -> CompState:
    """Build an initial state from raw input (dict / free-text) or a case folder.

    Pass ``case_dir`` to drive the pipeline from a ``valuation_case_XXX/`` folder;
    otherwise ``raw_input`` (a dict of fields or a free-text listing) is used.
    """
    state: CompState = {
        "raw_input": raw_input,
        "subject": {},
        "documents": {},
        "trace": [],
        "errors": [],
    }
    if case_dir:
        state["case_dir"] = case_dir
    return state
