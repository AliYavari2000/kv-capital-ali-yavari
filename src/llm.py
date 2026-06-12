"""LLM access with a deterministic fallback.

Two narrow LLM jobs in this agent:
  1. parse_listing  -- turn a messy free-text listing into structured fields.
  2. write_narrative -- turn the (already computed) valuation facts into a
     short underwriting rationale.

The LLM never produces numbers used in the valuation; it only parses input and
explains results. If ``OPENAI_API_KEY`` is unset (or the call fails) we fall
back to deterministic regex parsing and a templated narrative, so the whole
graph runs offline with zero external calls.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

from src import config

load_dotenv()

_MODEL = os.getenv("KV_LLM_MODEL", "gpt-4o-mini")


def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _client():
    from openai import OpenAI  # imported lazily so offline use needs no network
    return OpenAI()


# ---------------------------------------------------------------------------
# Listing parsing
# ---------------------------------------------------------------------------
_KNOWN_NB = list(config.NEIGHBORHOODS.keys())


def _fallback_parse(text: str) -> dict[str, Any]:
    t = text.lower()
    out: dict[str, Any] = {}

    m = re.search(r"(\d+)\s*(?:bed|bd|br|bedroom)", t)
    if m:
        out["bedrooms"] = int(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:bath|ba|bathroom)", t)
    if m:
        out["bathrooms"] = float(m.group(1))
    m = re.search(r"(\d[\d,]{2,5})\s*(?:sq\s?ft|sqft|sf|square\s?feet)", t)
    if m:
        out["gla_sqft"] = int(m.group(1).replace(",", ""))
    m = re.search(r"(?:built|year built|yr built|vintage)\D{0,8}(19\d{2}|20\d{2})", t)
    if m:
        out["year_built"] = int(m.group(1))
    m = re.search(r"lot\D{0,8}(\d[\d,]{2,6})", t)
    if m:
        out["lot_size_sqft"] = int(m.group(1).replace(",", ""))

    if re.search(r"semi[\s-]?detached", t):
        out["property_type"] = "Semi-Detached"
    elif re.search(r"town\s?house|town\s?home|row house", t):
        out["property_type"] = "Townhouse"
    elif re.search(r"condo|apartment|apt", t):
        out["property_type"] = "Condo"
    elif re.search(r"detached|single family|house", t):
        out["property_type"] = "Detached"

    for nb in _KNOWN_NB:
        if nb.lower() in t:
            out["neighborhood"] = nb
            break
    if "edmonton" in t:
        out["city"] = "Edmonton"
    elif "calgary" in t:
        out["city"] = "Calgary"

    return out


_PARSE_SYSTEM = (
    "You extract structured fields from a residential real-estate listing for an "
    "Alberta (Canada) mortgage underwriting tool. Return ONLY JSON with keys: "
    "address, city, neighborhood, property_type, bedrooms, bathrooms, gla_sqft, "
    "lot_size_sqft, year_built. Use null for anything not stated. "
    "property_type must be one of: Detached, Semi-Detached, Townhouse, Condo. "
    "gla_sqft and lot_size_sqft are integers in square feet. "
    f"Prefer neighborhood values from this list when applicable: {', '.join(_KNOWN_NB)}."
)


def parse_listing(text: str) -> dict[str, Any]:
    """Parse free-text into subject fields. LLM if available, else regex."""
    if not llm_available():
        return _fallback_parse(text)
    try:
        resp = _client().chat.completions.create(
            model=_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PARSE_SYSTEM},
                {"role": "user", "content": text},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        return {k: v for k, v in data.items() if v is not None}
    except Exception:
        return _fallback_parse(text)


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------
def _fallback_narrative(ctx: dict[str, Any]) -> str:
    s = ctx["subject"]
    v = ctx["valuation"]
    risk = ctx["risk"]
    n = v.get("comp_count", 0)
    flags = risk.get("flags", [])
    flag_txt = "; ".join(f["message"].rstrip(".") for f in flags) if flags else "no material risk flags"
    return (
        f"The subject is a {s.get('bedrooms','?')}-bed / {s.get('bathrooms','?')}-bath "
        f"{s.get('property_type','property')} of ~{s.get('gla_sqft','?')} sqft in "
        f"{s.get('neighborhood','the area')}, {s.get('city','AB')}. "
        f"Valuation reconciles {n} adjusted comparable sales to a point estimate of "
        f"${v.get('point_estimate',0):,.0f} (range ${v.get('low',0):,.0f}-${v.get('high',0):,.0f}, "
        f"implied ${v.get('implied_ppsf',0):,.0f}/sqft). Confidence is {risk.get('confidence','?')}: "
        f"{flag_txt}."
    )


_NARRATIVE_SYSTEM = (
    "You are a senior mortgage underwriter at KV Capital writing a concise, factual "
    "valuation rationale for a residential comp analysis. Use ONLY the numbers given; "
    "never invent figures. 4-7 sentences. Explain the estimate, the quality of the comps, "
    "the main adjustments, and any risks a credit committee should weigh."
)


def write_narrative(ctx: dict[str, Any]) -> str:
    """Write a short underwriting rationale from computed facts."""
    if not llm_available():
        return _fallback_narrative(ctx)
    try:
        resp = _client().chat.completions.create(
            model=_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _NARRATIVE_SYSTEM},
                {"role": "user", "content": json.dumps(ctx, default=str)},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return _fallback_narrative(ctx)
