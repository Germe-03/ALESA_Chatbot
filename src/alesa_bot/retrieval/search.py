# ===================== FILE: src/alesa_bot/retrieval/search.py =====================
from __future__ import annotations
import re, time
from pathlib import Path
from typing import List, Set
from src.alesa_bot.core.types import Hit
from .indexer import FileIndexer

STOPWORDS = {"und","oder","aber","dass","die","der","das","ein","eine","ist","sind","mit","auf","im","in","am","an","für","von","zu","zum","zur","wenn","wie","was"}


def _keywords(text: str) -> List[str]:
    words = re.findall(r"[A-Za-zÄÖÜäöüß0-9_-]{3,}", text.lower())
    return [w for w in words if w not in STOPWORDS]


class SimpleRetriever:
    def __init__(self, indexer: FileIndexer, time_limit_sec: int = 4) -> None:
        self.indexer = indexer
        self.time_limit_sec = time_limit_sec

    def _scan(self, path: Path, keys: Set[str]) -> List[Hit]:
        hits: List[Hit] = []
        for page_no, content in self.indexer.data.get(str(path), []):
            lower = content.lower()
            for k in keys:
                idx = lower.find(k)
                if idx != -1:
                    start = max(0, idx - 140)
                    end = min(len(content), idx + 140)
                    snippet = content[start:end].replace('\n', ' ')
                    hits.append(Hit(path=path, page=page_no, snippet=snippet))
        return hits

    def search(self, query: str, top_k: int = 6) -> List[Hit]:
        keys = set(_keywords(query))
        if not keys:
            return []
        t0 = time.time()
        hits: List[Hit] = []
        for path_str in self.indexer.data.keys():
            hits.extend(self._scan(Path(path_str), keys))
            if len(hits) >= top_k or (time.time() - t0) > self.time_limit_sec:
                break
        # deduplicate
        unique, seen = [], set()
        for h in hits:
            key = (str(h.path), h.page, h.snippet[:80])
            if key not in seen:
                seen.add(key)
                unique.append(h)
        return unique[:top_k]


#