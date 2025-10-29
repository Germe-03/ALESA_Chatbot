from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .indexer import FileIndexer


ARTICLE_RX = re.compile(r"\b(\d{4})[\._\-]?(\d{4})\b")  # 6042.0206 / 6042-0206 / 6042_0206 / 60420206


def normalize_article(s: str) -> str:
    if not s:
        return ""
    m = ARTICLE_RX.search(s)
    if not m:
        return re.sub(r"\D", "", s)
    return (m.group(1) + m.group(2)).strip()


HEADER_SYNONYMS: Dict[str, List[str]] = {
    "d1": ["d1", "D1"],
    "b": ["b", "B"],
    "b2": ["b2", "B2"],
    "nuttiefe": ["nuttiefe", "nutttiefe", "nut-tiefe", "nut tiefe"],
    "d2": ["d2", "D2"],
    "d3": ["d3", "D3"],
    "zahnform": ["zahnform", "zahn-form"],
    "aufnahme": ["aufnahme", "aufnahme-ø", "aufnahme ø", "aufnahme d"],
}


@dataclass
class ProductRow:
    code_raw: str
    code_norm: str
    d1: str = ""
    b: str = ""
    b2: str = ""
    nuttiefe: str = ""
    d2: str = ""
    d3: str = ""
    zahnform: str = ""
    aufnahme: str = ""
    source_path: Optional[Path] = None
    source_page: Optional[int] = None

    def as_row(self) -> List[str]:
        return [
            self.code_raw,
            self.d1, self.b, self.b2, self.nuttiefe, self.d2, self.d3, self.zahnform, self.aufnahme,
        ]


class ProductTableStore:
    """In-memory store of structured rows from CSV and heuristically parsed PDF pages."""

    def __init__(self) -> None:
        self.by_code: Dict[str, ProductRow] = {}

    # --------- Building ---------
    def ingest_csv(self, files: Iterable[Path]) -> int:
        count = 0
        for f in files:
            try:
                with f.open("r", encoding="utf-8", errors="ignore") as fp:
                    rd = csv.DictReader(fp)
                    for r in rd:
                        code_raw = (r.get("artikel") or r.get("Artikel") or r.get("code") or r.get("Code") or "").strip()
                        if not code_raw:
                            continue
                        code_norm = normalize_article(code_raw)
                        pr = ProductRow(
                            code_raw=code_raw,
                            code_norm=code_norm,
                            d1=(r.get("d1") or r.get("D1") or "").strip(),
                            b=(r.get("b") or r.get("B") or "").strip(),
                            b2=(r.get("b2") or r.get("B2") or "").strip(),
                            nuttiefe=(r.get("nuttiefe") or r.get("Nuttiefe") or "").strip(),
                            d2=(r.get("d2") or r.get("D2") or "").strip(),
                            d3=(r.get("d3") or r.get("D3") or "").strip(),
                            zahnform=(r.get("zahnform") or r.get("Zahnform") or "").strip(),
                            aufnahme=(r.get("aufnahme") or r.get("Aufnahme") or "").strip(),
                        )
                        self._upsert(pr)
                        count += 1
            except Exception:
                continue
        return count

    def ingest_from_indexer(self, indexer: FileIndexer) -> int:
        """Heuristically parse PDF page text to rows when possible.

        This is a best-effort approach for catalogs where text extraction preserves lines.
        """
        total = 0
        for path_str, pages in indexer.data.items():
            p = Path(path_str)
            for page_no, content in pages:
                rows = _extract_rows_from_page(content)
                for r in rows:
                    r.source_path = p
                    r.source_page = page_no
                    self._upsert(r)
                    total += 1
        return total

    def _upsert(self, row: ProductRow) -> None:
        if not row.code_norm:
            return
        prev = self.by_code.get(row.code_norm)
        if prev is None:
            self.by_code[row.code_norm] = row
            return
        # prefer row with more populated fields
        def _filled(x: ProductRow) -> int:
            return sum(1 for v in [x.d1, x.b, x.b2, x.nuttiefe, x.d2, x.d3, x.zahnform, x.aufnahme] if v)
        if _filled(row) > _filled(prev):
            self.by_code[row.code_norm] = row

    # --------- Query ---------
    def find_by_code(self, code: str) -> Optional[ProductRow]:
        n = normalize_article(code)
        return self.by_code.get(n)


def _detect_header_order(lines: List[str]) -> List[str]:
    best_idx, best_score, best_cols = -1, -1, []
    for i, line in enumerate(lines[:40]):
        low = line.lower()
        score = 0
        order: List[str] = []
        for token in re.split(r"\s{2,}|\t", low):
            token = token.strip()
            if not token:
                continue
            for canon, alts in HEADER_SYNONYMS.items():
                if any(a in token for a in alts):
                    if canon not in order:
                        order.append(canon)
                        score += 1
        if score >= 3 and score > best_score:
            best_idx, best_score, best_cols = i, score, order
    return best_cols


def _extract_rows_from_page(text: str) -> List[ProductRow]:
    if not text:
        return []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return []
    header_order = _detect_header_order(lines)
    rows: List[ProductRow] = []
    for line in lines:
        m = ARTICLE_RX.search(line)
        if not m:
            continue
        code_raw = m.group(0)
        code_norm = (m.group(1) + m.group(2))
        # split on 2+ spaces to better mimic table columns
        toks = re.split(r"\s{2,}|\t", line)
        # ensure first token is article, remove leading titles etc.
        # find the token that contains the article code
        art_idx = -1
        for j, t in enumerate(toks):
            if ARTICLE_RX.search(t):
                art_idx = j; break
        if art_idx < 0:
            continue
        after = toks[art_idx+1:]
        fields: Dict[str, str] = {k: "" for k in HEADER_SYNONYMS.keys()}
        for k, v in zip(header_order, after):
            fields[k] = v.strip()
        rows.append(ProductRow(
            code_raw=code_raw,
            code_norm=code_norm,
            d1=fields.get("d1", ""),
            b=fields.get("b", ""),
            b2=fields.get("b2", ""),
            nuttiefe=fields.get("nuttiefe", ""),
            d2=fields.get("d2", ""),
            d3=fields.get("d3", ""),
            zahnform=fields.get("zahnform", ""),
            aufnahme=fields.get("aufnahme", ""),
        ))
    return rows
