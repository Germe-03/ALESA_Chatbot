# ===================== FILE: src/alesa_bot/retrieval/indexer.py =====================
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
from src.alesa_bot.adapters.file_io import iter_files

try:
    from pypdf import PdfReader
except Exception:  # optional
    PdfReader = None


class FileIndexer:
    """Kleiner In-Memory-Index: {pfad: [(seite|None, text), ...]}"""
    def __init__(self, roots: List[Path], max_mb: int = 15, max_pdf_pages: int = 15) -> None:
        self.roots = roots
        self.max_mb = max_mb
        self.max_pdf_pages = max_pdf_pages
        self._index: Dict[str, List[Tuple[int | None, str]]] = {}

    def build(self) -> None:
        print("⏳ Baue Dateindex auf …", flush=True)
        for path in iter_files(self.roots):
            try:
                if path.stat().st_size > self.max_mb * 1024 * 1024:
                    continue
            except Exception:
                continue
            ext = path.suffix.lower()
            if ext in {'.txt', '.md'}:
                try:
                    text = path.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    text = ''
                if text:
                    self._index[str(path)] = [(None, text)]
            elif ext == '.pdf' and PdfReader is not None:
                try:
                    reader = PdfReader(str(path))
                    pages = []
                    total = min(len(reader.pages), self.max_pdf_pages)
                    for i in range(total):
                        try:
                            t = reader.pages[i].extract_text() or ''
                        except Exception:
                            t = ''
                        pages.append((i + 1, t))
                    if pages:
                        self._index[str(path)] = pages
                except Exception:
                    continue
        print(f"✅ Index aufgebaut: {len(self._index)} Dateien", flush=True)

    @property
    def data(self) -> Dict[str, List[Tuple[int | None, str]]]:
        return self._index


