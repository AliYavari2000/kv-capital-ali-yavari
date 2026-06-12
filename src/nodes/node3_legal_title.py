"""Node 3 - Legal / Title (``LegalTitleNode``).

Confirms the subject's legal identity and title: address, legal description,
parcel/linc ID, registered owner, and any title issues (liens, encumbrances).

In production this wraps a land-titles / parcel source. Here it derives a
deterministic, clearly-synthetic legal record from the normalized subject so the
workflow has a title step to reason about, and flags anything it cannot confirm.

Tools: land-title source, parcel lookup.
"""

from __future__ import annotations

import hashlib
from typing import Any

from src.state import CompState


def _stable_int(seed: str, mod: int) -> int:
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return int(h, 16) % mod


def _flag(code: str, severity: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def legal_title_node(state: CompState) -> dict[str, Any]:
    s = state.get("subject", {})
    address = s.get("address")
    city = s.get("city") or "AB"
    nb = s.get("neighborhood") or "Unknown"
    flags: list[dict[str, str]] = []

    seed = f"{address}|{nb}|{city}"
    # Synthetic Alberta-style parcel identifiers derived deterministically.
    linc = f"{_stable_int(seed + 'a', 9000) + 1000:04d}-{_stable_int(seed + 'b', 9000) + 1000:04d}-{_stable_int(seed + 'c', 90) + 10:02d}"
    plan = f"{_stable_int(seed + 'p', 8999999) + 1000000}"
    block = _stable_int(seed + 'blk', 40) + 1
    lot = _stable_int(seed + 'lot', 60) + 1
    legal_description = f"Plan {plan}, Block {block}, Lot {lot}"

    address_confirmed = bool(address)
    if not address_confirmed:
        flags.append(_flag("address_unconfirmed", "high",
                           "Subject address not provided; legal identity could not be confirmed."))

    # A small, deterministic share of subjects surface a title condition for the
    # workflow to handle (encumbrance / caveat).
    title_status = "Clear"
    if address_confirmed and _stable_int(seed + 'title', 100) < 12:
        title_status = "Encumbered"
        flags.append(_flag("title_encumbrance", "medium",
                           "Registered caveat/encumbrance on title; confirm it does not impair marketability."))

    legal_title = {
        "address_confirmed": address_confirmed,
        "legal_description": legal_description if address_confirmed else "Unverified",
        "parcel_id": linc if address_confirmed else "Unverified",
        "title_status": title_status if address_confirmed else "Unverified",
        "registered_owner": s.get("registered_owner", "On file with KV Capital"),
        "flags": flags,
    }

    trace = state.get("trace", []) + [
        f"legal_title: address_confirmed={address_confirmed}, parcel={legal_title['parcel_id']}, "
        f"title={legal_title['title_status']}, flags={len(flags)}"
    ]
    return {"legal_title": legal_title, "trace": trace}
