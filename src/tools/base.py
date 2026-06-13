"""Shared scaffolding for per-node agent toolkits.

Every LLM-agent node binds a toolkit (subclass of :class:`ToolkitBase`) whose
methods are exposed to the model as tools. The base provides the two tools every
node shares -- ``append_evidence`` and ``raise_human_review`` -- plus helpers for
building OpenAI function schemas concisely.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from src import data_sources as ds




def fn_spec(name: str, description: str,
            properties: dict[str, Any] | None = None,
            required: list[str] | None = None) -> dict[str, Any]:
    """Build an OpenAI function (tool) schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }



# Shared tool schemas (same on every node).
EVIDENCE_SPEC = fn_spec(
    "append_evidence",
    "Log a source document or fact into the evidence trail for this node. "
    "Prefer source_id from list_data_sources when citing Calgary/Alberta authorities.",
    {"source": {"type": "string"},
     "detail": {"type": "string"},
     "source_id": {"type": "string", "description": "Registry id, e.g. open_calgary_assessment"}},
    ["source"],
)
LIST_SOURCES_SPEC = fn_spec(
    "list_data_sources",
    "List the Calgary/Alberta authoritative data sources this node is expected to use.",
)
REVIEW_SPEC = fn_spec(
    "raise_human_review",
    "Escalate to a human reviewer when something is ambiguous, conflicting, or missing.",
    {"reason": {"type": "string"},
     "severity": {"type": "string", "enum": ["low", "medium", "high"]}},
    ["reason"],
)

SHARED_TOOL_SPECS = [EVIDENCE_SPEC, REVIEW_SPEC, LIST_SOURCES_SPEC]


class ToolkitBase:
    """Common context + shared tools for an agent node."""

    node_name = "node"

    def __init__(self) -> None:
        self.evidence: list[dict[str, Any]] = []
        self.escalations: list[dict[str, Any]] = []

    # -- shared tools ----------------------------------------------------
    def append_evidence(self, source: str, detail: str = "",
                        source_id: str = "") -> dict[str, Any]:
        entry: dict[str, Any] = {
            "node": self.node_name,
            "source": source,
            "detail": detail,
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        if source_id:
            src = ds.get_source(source_id)
            entry["source_id"] = source_id
            entry["provider"] = src.get("provider")
            entry["authority"] = src.get("name")
            if not detail:
                entry["detail"] = src.get("why_it_matters", "")
        self.evidence.append(entry)
        return {"logged": True, "evidence_count": len(self.evidence)}

    def list_data_sources(self) -> dict[str, Any]:
        """Return authoritative Calgary/Alberta sources for this node."""
        sources = ds.sources_for_node(self.node_name)
        return {"jurisdiction": ds.JURISDICTION, "node": self.node_name, "sources": sources}

    def raise_human_review(self, reason: str, severity: str = "medium") -> dict[str, Any]:
        sev = severity if severity in ("low", "medium", "high") else "medium"
        self.escalations.append({"reason": reason, "severity": sev})
        return {"escalated": True, "escalation_count": len(self.escalations)}

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _any_exists(values: Any) -> bool:
        """True if any value (a document-hook dict, or a list of hook dicts) exists."""
        for v in values:
            if isinstance(v, dict) and v.get("exists"):
                return True
            if isinstance(v, list) and any(isinstance(x, dict) and x.get("exists") for x in v):
                return True
        return False

    def shared_dispatch(self) -> dict[str, Any]:
        return {
            "append_evidence": self.append_evidence,
            "raise_human_review": self.raise_human_review,
            "list_data_sources": self.list_data_sources,
        }

    def shared_specs(self) -> list[dict[str, Any]]:
        return list(SHARED_TOOL_SPECS)
