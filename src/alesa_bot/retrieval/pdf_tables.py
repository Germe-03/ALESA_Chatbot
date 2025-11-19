from __future__ import annotations

"""
Robuste Tabellenextraktion aus PDFs mittels pdfplumber.
Ziel: Produktzeilen strukturiert erfassen, unabhängig vom Tabellenlayout.

Abhängig von pdfplumber (optional). Wenn nicht installiert, werden keine
Zeilen extrahiert (Aufrufer soll das abfangen).
"""

from pathlib import Path
from typing import List, Dict, Iterable
import re

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pdfplumber = None  # type: ignore

from .tables import ProductRow, HEADER_SYNONYMS, ARTICLE_RX, normalize_article


def _nz_numeric(s: str) -> str:
    """Local helper: return 'NULL' if empty for numeric-like cells (SQL-style)."""
    if s is None:
        return "NULL"
    s2 = s.strip()
    return s2 if s2 else "NULL"


def _header_map(row: Iterable[str]) -> List[str]:
    """Versucht aus einer Tabellenkopf‑Zeile die kanonischen Spaltennamen
    abzuleiten (d1, b, b2, nuttiefe, d2, d3, zahnform, aufnahme).
    Gibt die erkannte Reihenfolge zurück (leere Strings für unbekannte Spalten)."""
    order: List[str] = []
    for cell in row:
        token = (cell or "").strip().lower()
        found = ""
        for canon, alts in HEADER_SYNONYMS.items():
            if any(a in token for a in alts):
                found = canon
                break
        order.append(found)
    return order


def _rows_from_matrix(mat: List[List[str]], page_no: int, path: Path) -> List[ProductRow]:
    rows: List[ProductRow] = []
    if not mat:
        return rows
    # pick the first row that yields >=3 matched headers as header
    header_idx = -1
    header_order: List[str] = []
    for i, r in enumerate(mat[:5]):
        m = _header_map(r)
        if sum(1 for x in m if x) >= 3:
            header_idx = i
            header_order = m
            break
    start = header_idx + 1 if header_idx >= 0 else 0

    for r in mat[start:]:
        # find article code in any column
        art_idx = -1
        code_raw = ""
        for j, c in enumerate(r):
            m = ARTICLE_RX.search(c or "")
            if m:
                art_idx = j
                code_raw = m.group(0)
                break
        if art_idx < 0:
            continue
        code_norm = normalize_article(code_raw)
        # map fields according to header (if we have it)
        fields: Dict[str, str] = {k: "" for k in HEADER_SYNONYMS.keys()}
        if header_order:
            for k, v in zip(header_order, r):
                if k:
                    fields[k] = (v or "").strip()
        else:
            # Fallback: try heuristic positions relative to article column
            def at(offset: int) -> str:
                idx = art_idx + offset
                return (r[idx] if 0 <= idx < len(r) else "").strip()

            fields.update({
                "d1": at(1),
                "b": at(2),
                "b2": at(3),
                "nuttiefe": "",
                "d2": "",
                "d3": "",
                "zahnform": "",
                "aufnahme": "",
            })

        rows.append(ProductRow(
            code_raw=code_raw,
            code_norm=code_norm,
            d1=_nz_numeric(fields.get("d1", "")),
            b=_nz_numeric(fields.get("b", "")),
            b2=_nz_numeric(fields.get("b2", "")),
            nuttiefe=_nz_numeric(fields.get("nuttiefe", "")),
            d2=_nz_numeric(fields.get("d2", "")),
            d3=_nz_numeric(fields.get("d3", "")),
            zahnform=fields.get("zahnform", ""),
            aufnahme=_nz_numeric(fields.get("aufnahme", "")),
            source_path=path,
            source_page=page_no,
        ))
    return rows


def extract_rows_from_pdf(path: Path) -> List[ProductRow]:
    if pdfplumber is None:
        return []
    out: List[ProductRow] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    # try table extraction; fallback to text boxes if needed
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                for tbl in tables:
                    # clean rows: ensure strings
                    mat = [[(c if isinstance(c, str) else (c or "")) for c in row] for row in (tbl or [])]
                    out.extend(_rows_from_matrix(mat, i, path))
                # Fallback: try settings that are more text-oriented
                if not tables:
                    try:
                        tables2 = page.extract_tables(table_settings={
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "snap_tolerance": 3,
                        }) or []
                    except Exception:
                        tables2 = []
                    for tbl in tables2:
                        mat = [[(c if isinstance(c, str) else (c or "")) for c in row] for row in (tbl or [])]
                        out.extend(_rows_from_matrix(mat, i, path))
    except Exception:
        return []
    return out
