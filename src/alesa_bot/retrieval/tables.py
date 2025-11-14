from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .indexer import FileIndexer
try:
    # optional pdf table extraction
    from .pdf_tables import extract_rows_from_pdf  # type: ignore
except Exception:
    def extract_rows_from_pdf(path: Path):  # type: ignore
        return []


ARTICLE_RX = re.compile(r"\b(\d{4})[\._\-]?(\d{4})\b")  # 6042.0206 / 6042-0206 / 6042_0206 / 60420206


def normalize_article(s: str) -> str:
    if not s:
        return ""
    m = ARTICLE_RX.search(s)
    if not m:
        return re.sub(r"\D", "", s)
    return (m.group(1) + m.group(2)).strip()


def _nz_numeric(s: str) -> str:
    """Return 'NULL' (SQL-style) when a numeric cell is empty.

    Used for numeric-like fields: d1, b, b2, nuttiefe, d2, d3, aufnahme.
    """
    if s is None:
        return "NULL"
    s2 = s.strip()
    return s2 if s2 else "NULL"


HEADER_SYNONYMS: Dict[str, List[str]] = {
    # Außendurchmesser
    "d1": [
        "d1", "d 1", "d-1", "D1", "ø d1", "ø d 1", "ø d-1", "ø d", "ød",
        "durchmesser", "aussendurchmesser", "außendurchmesser", "außen-ø", "aussen-ø", "außen ø",
        "ø", "diameter", "D"
    ],
    # Schnitt-/Körperbreite
    "b": ["b", "B", "schnittbreite", "schnitt-dicke", "schnittdicke", "kerf", "breite"],
    "b2": ["b2", "B2", "körperdicke", "koerperdicke", "blattdicke", "koerper", "körper"],
    # Nuten
    "nuttiefe": ["nuttiefe", "nut-tiefe", "nut tiefe", "nutttiefe", "nutteife", "nuttiefe t", "t", "t1", "h"],
    # Zusatzdurchmesser
    "d2": ["d2", "D2"],
    "d3": ["d3", "D3"],
    # Zähne
    "zahnform": ["zahnform", "zahn-form", "form", "zähneform"],
    # Aufnahme/Bohrung
    "aufnahme": [
        "aufnahme", "aufnahme-ø", "aufnahme ø", "aufnahme d", "bohrung", "bohr-ø", "bohrung d",
        "bohrungsdurchmesser", "innen-ø", "innen ø", "bohr Ø"
    ],
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
        """Ingest CSVs with robust delimiter and header normalization.

        - Detects delimiter (comma vs semicolon)
        - Normalizes headers like "ArtikelNr.", "d1 mm", "b1 mm", "Aufnahmen" to
          canonical keys: artikel, d1, b, b2, nuttiefe, d2, d3, zahnform, aufnahme
        - Ignores unknown columns
        """
        def guess_delim(sample: str) -> str:
            sc, cc = sample.count(";"), sample.count(",")
            # Prefer semicolon if clearly more ; than , in the header line
            if sc > cc:
                return ";"
            return ","

        def canonize_header(h: str) -> str:
            if not h:
                return ""
            s = h.strip().lower()
            # remove units and punctuation/spaces/umlauts variants
            s = s.replace(" mm", "").replace(" (mm)", "")
            s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
            s = s.replace("/", " ").replace("-", " ")
            s = s.replace(".", "").replace(":", "").replace("\t", " ")
            s = " ".join(s.split())
            # simple mappings
            if s in ("artikelnr", "artikel nr", "artikelnummer", "artikel", "code"):
                return "artikel"
            if s in ("d1", "d 1", "d-1", "durchmesser", "aussendurchmesser", "aussen d", "d"):
                return "d1"
            if s in ("b", "schnittbreite", "schnitt dicke", "schnittdicke", "breite", "kerf"):
                return "b"
            if s in ("b1", "b2", "blattdicke", "koerper", "koerperdicke", "koerper dicke"):
                return "b2"
            if s in ("nuttiefe", "nut tiefe", "t", "t1", "h"):
                return "nuttiefe"
            if s in ("d2",):
                return "d2"
            if s in ("d3",):
                return "d3"
            if s in ("zahnform", "form"):
                return "zahnform"
            if s in ("aufnahme", "bohrung", "bohr d", "bohrungsdurchmesser", "innen d", "aufnahmen"):
                return "aufnahme"
            return ""

        count = 0
        for f in files:
            try:
                with f.open("r", encoding="utf-8", errors="ignore") as fp:
                    # Read header line to determine delimiter and canonical field names
                    first = fp.readline()
                    if not first:
                        continue
                    delim = guess_delim(first)
                    headers_raw = [h.strip() for h in first.strip().split(delim)]
                    headers = [canonize_header(h) for h in headers_raw]
                    # Build index map for needed fields
                    idx: Dict[str, int] = {}
                    for i, h in enumerate(headers):
                        if h and h not in idx:
                            idx[h] = i
                    # iterate remaining lines
                    for line in fp:
                        if not line or not line.strip() or set(line.strip()) == set(";,"):
                            continue
                        parts = [p.strip() for p in line.rstrip("\n\r").split(delim)]
                        # pad parts to length of header for safe indexing
                        if len(parts) < len(headers):
                            parts += [""] * (len(headers) - len(parts))
                        # fetch fields
                        def getv(key: str) -> str:
                            j = idx.get(key)
                            return parts[j].strip() if j is not None else ""

                        code_raw = getv("artikel")
                        if not code_raw:
                            # try loose: find any token in the row that looks like an article number
                            merged = " ".join(parts)
                            m = ARTICLE_RX.search(merged)
                            code_raw = m.group(0) if m else ""
                        if not code_raw:
                            continue
                        code_norm = normalize_article(code_raw)
                        pr = ProductRow(
                            code_raw=code_raw,
                            code_norm=code_norm,
                            d1=_nz_numeric(getv("d1")),
                            b=_nz_numeric(getv("b")),
                            b2=_nz_numeric(getv("b2")),
                            nuttiefe=_nz_numeric(getv("nuttiefe")),
                            d2=_nz_numeric(getv("d2")),
                            d3=_nz_numeric(getv("d3")),
                            zahnform=getv("zahnform"),
                            aufnahme=_nz_numeric(getv("aufnahme")),
                        )
                        self._upsert(pr)
                        count += 1
            except Exception:
                # ignore file-level errors to keep startup resilient
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

    def count(self) -> int:
        return len(self.by_code)

    def ingest_from_pdf_files(self, files: Iterable[Path]) -> int:
        """Extrahiert Tabellen strukturiert mit pdfplumber (sofern installiert)."""
        total = 0
        for f in files:
            if f.suffix.lower() != ".pdf":
                continue
            try:
                rows = extract_rows_from_pdf(f)
            except Exception:
                rows = []
            for r in rows:
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
            def _is_val(v: str) -> bool:
                if not v:
                    return False
                vs = v.strip()
                if not vs:
                    return False
                # Treat SQL-style NULL as empty
                if vs.upper() == "NULL":
                    return False
                return True
            return sum(1 for v in [x.d1, x.b, x.b2, x.nuttiefe, x.d2, x.d3, x.zahnform, x.aufnahme] if _is_val(v))
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
            d1=_nz_numeric(fields.get("d1", "")),
            b=_nz_numeric(fields.get("b", "")),
            b2=_nz_numeric(fields.get("b2", "")),
            nuttiefe=_nz_numeric(fields.get("nuttiefe", "")),
            d2=_nz_numeric(fields.get("d2", "")),
            d3=_nz_numeric(fields.get("d3", "")),
            zahnform=fields.get("zahnform", ""),
            aufnahme=_nz_numeric(fields.get("aufnahme", "")),
        ))
    return rows
