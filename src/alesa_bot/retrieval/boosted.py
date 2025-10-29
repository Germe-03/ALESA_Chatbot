from __future__ import annotations

import re
import time
from typing import List, Set
from pathlib import Path

from src.alesa_bot.core.types import Hit
from .indexer import FileIndexer


ARTICLE_RX = re.compile(r"\b\d{4}(?:[\._\-]?\d{4})\b")  # 6042.0206 / 6042-0206 / 6042_0206 / 60420206


class BoostedRetriever:
    """
    Wrapper that prioritizes exact keyword hits (e.g., article numbers) from the raw
    index before falling back to a base semantic retriever.
    """

    def __init__(self, indexer: FileIndexer, base, time_limit_sec: int = 4):
        self.indexer = indexer
        self.base = base
        self.time_limit_sec = time_limit_sec

    def _keyword_hits(self, query: str, top_k: int) -> List[Hit]:
        t0 = time.time()
        q = (query or "").strip()
        hits: List[Hit] = []

        # Detect article code patterns
        codes: Set[str] = set(m.group(0) for m in ARTICLE_RX.finditer(q))
        if not codes:
            return []

        for path_str, pages in self.indexer.data.items():
            p = Path(path_str)
            for page_no, content in pages:
                lower = (content or "").lower()
                for code in codes:
                    if code.lower() in lower:
                        snippet = content
                        if len(snippet) > 320:
                            idx = lower.find(code.lower())
                            start = max(0, idx - 160)
                            end = min(len(content), idx + 160)
                            snippet = content[start:end]
                        hits.append(Hit(path=p, page=page_no, snippet=snippet.replace("\n", " ").strip()))
                        break
                if len(hits) >= top_k or (time.time() - t0) > self.time_limit_sec:
                    return hits[:top_k]
        return hits[:top_k]

    def search(self, query: str, top_k: int = 6) -> List[Hit]:
        prim = self._keyword_hits(query, top_k)
        if len(prim) >= top_k:
            return prim[:top_k]

        rest = self.base.search(query, top_k=top_k)
        # Deduplicate by path/page/snippet prefix
        out: List[Hit] = []
        seen = set()
        for h in prim + rest:
            key = (str(h.path), h.page, h.snippet[:80])
            if key in seen:
                continue
            seen.add(key)
            out.append(h)
            if len(out) >= top_k:
                break
        return out
