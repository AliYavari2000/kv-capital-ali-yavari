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


class LLMUnavailableError(RuntimeError):
    """Raised when an LLM agent node runs without an API key configured."""


# ---------------------------------------------------------------------------
# Tool-calling agent loop
# ---------------------------------------------------------------------------
def run_tool_agent(
    system_prompt: str,
    user_prompt: str,
    tool_specs: list[dict[str, Any]],
    dispatch: dict[str, Any],
    *,
    max_steps: int = 16,
    model: str | None = None,
) -> dict[str, Any]:
    """Drive an LLM agent that can call the supplied tools.

    The model is given ``tool_specs`` and decides which tools (in ``dispatch``)
    to call and in what order. Each tool's return value is fed back as a tool
    message until the model stops calling tools or ``max_steps`` is reached.

    There is no offline fallback: if no API key is configured this raises
    :class:`LLMUnavailableError`.
    """
    if not llm_available():
        raise LLMUnavailableError(
            "This node is an LLM agent and requires OPENAI_API_KEY to be set "
            "(no deterministic fallback)."
        )

    client = _client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    calls: list[dict[str, Any]] = []

    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=model or _MODEL,
            temperature=0,
            messages=messages,
            tools=tool_specs,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        if not tool_calls:
            break

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            fn = dispatch.get(name)
            if fn is None:
                result: Any = {"error": f"unknown tool '{name}'"}
            else:
                try:
                    result = fn(**args)
                except Exception as exc:  # surface tool errors back to the model
                    result = {"error": f"{type(exc).__name__}: {exc}"}
            calls.append({"tool": name, "args": args})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    final = messages[-1].get("content") if messages and messages[-1]["role"] == "assistant" else None
    return {"calls": calls, "messages": messages, "final": final}


def run_node_agent(
    state: dict[str, Any],
    dispatch: dict[str, Any],
    tool_specs: list[dict[str, Any]],
    system_prompt: str,
    user_prompt: str,
    fast_run: Any,
    *,
    max_steps: int = 16,
) -> tuple[dict[str, Any], str]:
    """Run an agent node: deterministic script (fast path) or LLM tool loop."""
    from src import fast_path

    if fast_path.enabled(state):
        return fast_run(), "script"
    return run_tool_agent(
        system_prompt, user_prompt, tool_specs, dispatch, max_steps=max_steps
    ), "LLM agent"


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
    from src import fast_path

    if fast_path.fast_narratives() or not llm_available():
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


# ---------------------------------------------------------------------------
# Reconciliation explanation (LLM explains the already-computed result)
# ---------------------------------------------------------------------------
_RECONCILIATION_SYSTEM = (
    "You are a senior residential appraiser writing the reconciliation section of "
    "a valuation report. You are given the ALREADY-COMPUTED final value, range, "
    "per-comp weights, adjusted prices, and bracketing result. Explain -- in 3-6 "
    "sentences of appraisal-style prose -- why the reconciled value is supported: "
    "which comps carried the most weight and why, whether the comps bracket the "
    "subject, and any caveats. Use ONLY the numbers provided; never compute or "
    "invent a different value."
)


def write_reconciliation(ctx: dict[str, Any]) -> str:
    """LLM explanation of the deterministic reconciliation. Requires an API key unless KV_FAST_NARRATIVE=1."""
    from src import fast_path

    if fast_path.fast_narratives():
        v = ctx["valuation"]
        n = len(ctx.get("comps", []))
        flags = ctx.get("flags") or []
        flag_txt = "; ".join(flags) if flags else "no material flags"
        return (
            f"The reconciled value of ${v.get('point_estimate', 0):,.0f} is supported by "
            f"{n} adjusted comparable sales (range ${v.get('low', 0):,.0f}–"
            f"${v.get('high', 0):,.0f}). Weighting favors the most similar, "
            f"least-adjusted comps. Bracketed: {ctx.get('bracketed')}. {flag_txt}."
        )
    if not llm_available():
        raise LLMUnavailableError(
            "Reconciliation narrative requires OPENAI_API_KEY (no fallback)."
        )
    resp = _client().chat.completions.create(
        model=_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _RECONCILIATION_SYSTEM},
            {"role": "user", "content": json.dumps(ctx, default=str)},
        ],
    )
    return resp.choices[0].message.content.strip()
