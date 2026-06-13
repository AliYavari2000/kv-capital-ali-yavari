"""Case-folder ingestion scaffold.

A "case" is a single appraisal assignment delivered as a structured folder
(``valuation_case_001/`` and friends). This module is the single seam that turns
that folder into something the rest of the agent can consume:

- **Structured formats** (``.csv`` / ``.json`` / ``.geojson`` / ``.md``) are
  parsed for real, right now.
- **Heavy formats** (``.pdf`` / ``.jpg`` / ``.tif`` / ``.docx`` / ``.xlsx``)
  are exposed as typed *document hooks* -- ``{"path", "kind", "parsed": False,
  ...}`` -- so the per-node tools (PDF text extraction, OCR, etc.) can slot in
  later without any caller changing.

The folder->node mapping lives in ``config.CASE_LAYOUT``. Sections 03
(assessment/tax) and 05 (permits/surveys) have no dedicated node; their accessors
are folded into ``legal_title()`` and ``zoning()`` respectively.

Nothing here computes a valuation; it only ingests and normalizes inputs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from src import config
from src import data_sources as ds

# Extension -> the hook "kind" reported to downstream tools.
_HOOK_KINDS = {
    ".pdf": "pdf",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".tif": "tif",
    ".tiff": "tif",
    ".docx": "docx",
    ".doc": "docx",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".zip": "archive",
}


def _norm_header(name: str) -> str:
    """Lowercase a header and strip spaces/underscores/hyphens for matching."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


# Precompute: normalized-alias -> canonical column name.
_ALIAS_INDEX: dict[str, str] = {}
for _canon, _aliases in config.COLUMN_ALIASES.items():
    _ALIAS_INDEX[_norm_header(_canon)] = _canon
    for _a in _aliases:
        _ALIAS_INDEX[_norm_header(_a)] = _canon


def normalize_comp_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map arbitrary case-CSV headers onto the canonical comp schema.

    Unknown columns are preserved as-is (so nothing is silently lost); known
    aliases are renamed to their canonical name. Numeric-looking values for the
    canonical numeric columns are coerced.
    """
    out: list[dict[str, Any]] = []
    numeric_int = {"bedrooms", "gla_sqft", "lot_size_sqft", "year_built"}
    numeric_float = {"lat", "lon", "bathrooms", "sale_price"}
    for row in rows:
        rec: dict[str, Any] = {}
        for key, value in row.items():
            canon = _ALIAS_INDEX.get(_norm_header(key))
            target = canon or key
            # Don't let an unknown column clobber a canonical one already set.
            if target in rec and canon is None:
                continue
            rec[target] = value
        for col in numeric_int:
            if rec.get(col) not in (None, ""):
                rec[col] = _to_int(rec[col])
        for col in numeric_float:
            if rec.get(col) not in (None, ""):
                rec[col] = _to_float(rec[col])
        out.append(rec)
    return out


def _to_int(v: Any) -> Optional[int]:
    try:
        return int(round(float(str(v).replace(",", "").replace("$", "").strip())))
    except (TypeError, ValueError):
        return None


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(str(v).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Document hook
# ---------------------------------------------------------------------------
def document_hook(path: str, *, exists: Optional[bool] = None,
                  section: str = "", file_key: str = "") -> dict[str, Any]:
    """Describe a not-yet-parsed file as a typed hook.

    The per-node tools (PDF text extraction, OCR, geo, etc.) replace the
    ``parsed``/``content`` fields without changing this contract. When
    ``section`` and ``file_key`` are supplied, the hook is annotated with the
    Calgary/Alberta authoritative source from ``data_sources``.
    """
    ext = os.path.splitext(path)[1].lower()
    if exists is None:
        exists = os.path.exists(path)
    hook: dict[str, Any] = {
        "path": path,
        "kind": _HOOK_KINDS.get(ext, "unknown"),
        "ext": ext,
        "exists": bool(exists),
        "parsed": False,
        "content": None,
    }
    if section and file_key:
        hook = ds.annotate_hook(hook, section, file_key)
    return hook


class Case:
    """Read-only view over a single case folder.

    Accessors return ``{"data": {...}, "documents": {...}}`` where ``data`` holds
    parsed structured inputs and ``documents`` holds typed hooks for the heavy
    files awaiting their tools.
    """

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Case folder not found: {self.root}")

    # -- path helpers ----------------------------------------------------
    def section_dir(self, section: str) -> str:
        rel = config.CASE_LAYOUT[section]["dir"]
        return os.path.join(self.root, rel)

    def path(self, section: str, filename: str) -> str:
        return os.path.join(self.section_dir(section), filename)

    # -- real readers ----------------------------------------------------
    def read_csv(self, abs_path: str) -> list[dict[str, Any]]:
        if not os.path.exists(abs_path):
            return []
        import pandas as pd

        df = pd.read_csv(abs_path)
        return df.to_dict(orient="records")

    def read_json(self, abs_path: str) -> Any:
        if not os.path.exists(abs_path):
            return None
        with open(abs_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def read_geojson(self, abs_path: str) -> Any:
        return self.read_json(abs_path)

    def read_text(self, abs_path: str) -> Optional[str]:
        if not os.path.exists(abs_path):
            return None
        with open(abs_path, "r", encoding="utf-8") as fh:
            return fh.read()

    # -- hook readers (deferred to per-node tools) -----------------------
    def read_pdf(self, abs_path: str) -> dict[str, Any]:
        return document_hook(abs_path)

    def read_image(self, abs_path: str) -> dict[str, Any]:
        return document_hook(abs_path)

    def read_tif(self, abs_path: str) -> dict[str, Any]:
        return document_hook(abs_path)

    def read_docx(self, abs_path: str) -> dict[str, Any]:
        return document_hook(abs_path)

    def read_xlsx(self, abs_path: str) -> dict[str, Any]:
        return document_hook(abs_path)

    def list_dir_documents(self, abs_dir: str, *, section: str = "",
                           file_key: str = "") -> list[dict[str, Any]]:
        """Return document hooks for every file under a section subfolder."""
        if not os.path.isdir(abs_dir):
            return []
        hooks: list[dict[str, Any]] = []
        for name in sorted(os.listdir(abs_dir)):
            full = os.path.join(abs_dir, name)
            if os.path.isfile(full):
                hooks.append(document_hook(full, section=section, file_key=file_key))
        return hooks

    # -- generic section loader -----------------------------------------
    def _read_data_file(self, abs_path: str) -> Any:
        ext = os.path.splitext(abs_path)[1].lower()
        if ext == ".csv":
            return self.read_csv(abs_path)
        if ext == ".json":
            return self.read_json(abs_path)
        if ext == ".geojson":
            return self.read_geojson(abs_path)
        if ext in (".md", ".txt"):
            return self.read_text(abs_path)
        return document_hook(abs_path)

    def _load_section(self, section: str) -> dict[str, Any]:
        spec = config.CASE_LAYOUT.get(section, {})
        sect_dir = self.section_dir(section)

        data: dict[str, Any] = {}
        sources: dict[str, Any] = {}
        for key, filename in spec.get("data", {}).items():
            data[key] = self._read_data_file(os.path.join(sect_dir, filename))
            sources[key] = ds.source_for_case_file(section, key)

        documents: dict[str, Any] = {}
        for key, filename in spec.get("documents", {}).items():
            documents[key] = document_hook(
                os.path.join(sect_dir, filename), section=section, file_key=key)
            sources[key] = ds.source_for_case_file(section, key)
        for key, subdir in spec.get("document_dirs", {}).items():
            documents[key] = self.list_dir_documents(
                os.path.join(sect_dir, subdir), section=section, file_key=key)
            sources[key] = ds.source_for_case_file(section, key)

        return {"data": data, "documents": documents, "sources": sources, "dir": sect_dir}

    # -- per-node accessors ---------------------------------------------
    def assignment(self) -> dict[str, Any]:
        return self._load_section("assignment")

    def subject(self) -> dict[str, Any]:
        return self._load_section("subject")

    def legal_title(self) -> dict[str, Any]:
        """02 legal/title plus 03 assessment/tax, folded together."""
        legal = self._load_section("legal_title")
        assessment = self._load_section("assessment_tax")
        legal["data"]["assessment"] = assessment["data"]
        legal["documents"]["assessment"] = assessment["documents"]
        legal["sources"] = {**legal.get("sources", {}),
                            **{f"assessment/{k}": v for k, v in assessment.get("sources", {}).items()}}
        return legal

    def zoning(self) -> dict[str, Any]:
        """04 zoning/land-use plus 05 permits/rpr/surveys, folded together."""
        zoning = self._load_section("zoning")
        permits = self._load_section("permits")
        zoning["data"]["permits"] = permits["data"]
        zoning["documents"]["permits"] = permits["documents"]
        zoning["sources"] = {**zoning.get("sources", {}),
                             **{f"permits/{k}": v for k, v in permits.get("sources", {}).items()}}
        return zoning

    def comps(self) -> dict[str, Any]:
        return self._load_section("comparables")

    def market(self) -> dict[str, Any]:
        return self._load_section("market")

    # -- output locations (populated by later nodes/tools) --------------
    def outputs_dir(self, *, create: bool = False) -> str:
        d = self.section_dir("workflow_outputs")
        if create:
            os.makedirs(d, exist_ok=True)
        return d

    def final_package_dir(self, *, create: bool = False) -> str:
        d = self.section_dir("final_package")
        if create:
            os.makedirs(d, exist_ok=True)
        return d

    # -- convenience -----------------------------------------------------
    def document_index(self) -> dict[str, Any]:
        """A flat-ish index of every section's data + document hooks, suitable
        for stashing on the graph state for audit/inspection."""
        index: dict[str, Any] = {"data_source_manifest": ds.case_manifest()}
        for section in config.CASE_LAYOUT:
            try:
                index[section] = self._load_section(section)
            except Exception as exc:  # pragma: no cover - defensive
                index[section] = {"error": str(exc)}
        return index

    def read_manifest(self) -> dict[str, Any]:
        """Read ``data_source_manifest.json`` from the case root, or build one."""
        path = os.path.join(self.root, "data_source_manifest.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return ds.case_manifest()


def load_case(case_dir: str) -> Case:
    """Resolve a case folder into a :class:`Case` view.

    ``case_dir`` may be absolute or relative to the project root.
    """
    if not os.path.isabs(case_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(project_root, case_dir)
        case_dir = candidate if os.path.isdir(candidate) else case_dir
    return Case(case_dir)
