from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from src.alesa_bot.core.types import Hit
from .indexer import FileIndexer
from .chunker import explode_pages, Chunk
from .embeddings import EmbeddingEncoder


def _tokenize(t: str) -> List[str]:
    return re.findall(r"[A-Za-zÄÖÜäöüß0-9_-]{2,}", t.lower())


# einfache deutsche Stopwörter (für Overlap-Logik)
STOPWORDS: Set[str] = {
    "und", "oder", "aber", "dass", "die", "der", "das", "ein", "eine", "ist", "sind", "war", "waren",
    "mit", "ohne", "auf", "im", "in", "am", "an", "für", "von", "zu", "zum", "zur", "wenn", "wie", "was",
    "es", "gibt", "einen", "eine", "den", "dem", "der", "des", "so", "auch", "z.b", "z", "b"
}

# Ranking-Feintuning
PATH_BOOSTS: Dict[str, float] = {
    "datenschutz": 1.25,
    "privacy": 1.25,
    "legal": 1.10,
    "agb": 1.02,  # minimaler Boost, damit AGB nicht alles dominiert
}

# Mindest-Overlap: mindestens 1 inhaltsreiches Token (Stopwörter ignoriert)
MIN_OVERLAP_CONTENT_TOKENS = 1


@dataclass
class Scored:
    chunk: Chunk
    score: float


class HybridRetriever:
    """
    Hybrid-Retriever:
      - Chunks mit Überlappung
      - Embeddings (Cosine)
      - Lexikalisch (BM25-lite)
      - Overlap-Filter (ohne Stopwörter) & Pfad-Boost
    """
    def __init__(
        self,
        indexer: FileIndexer,
        encoder: EmbeddingEncoder,
        time_limit_sec: int = 4,
        chunk_size: int = 800,
        overlap: int = 200,
    ) -> None:
        self.indexer = indexer
        self.encoder = encoder
        self.time_limit_sec = time_limit_sec
        self.chunk_size = chunk_size
        self.overlap = overlap

        self._chunk_bank: List[Chunk] = []
        self._vecs: List[List[float]] = []
        self._bow: List[Dict[str, int]] = []
        self._idf: Dict[str, float] = {}
        self._built = False

    def _build_bank(self) -> None:
        if self._built:
            return

        # 1) Seiten zu Chunks aufspalten
        for path_str, pages in self.indexer.data.items():
            path = Path(path_str)
            self._chunk_bank.extend(
                explode_pages(path, pages, size=self.chunk_size, overlap=self.overlap)
            )

        # 2) Embeddings
        texts = [c.text for c in self._chunk_bank]
        if texts:
            self._vecs = self.encoder.encode(texts)
        else:
            self._vecs = []

        # 3) BM25-lite (idf + bow)
        N = max(1, len(self._chunk_bank))
        df: Dict[str, int] = {}
        self._bow = []
        for c in self._chunk_bank:
            toks = set(_tokenize(c.text))
            bow: Dict[str, int] = {t: 1 for t in toks}
            self._bow.append(bow)
            for t in toks:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0) for t, df_t in df.items()}

        self._built = True

    @staticmethod
    def _cos(a: List[float], b: List[float]) -> float:
        num = sum(x * y for x, y in zip(a, b))
        da = math.sqrt(sum(x * x for x in a))
        db = math.sqrt(sum(y * y for y in b))
        if da == 0.0 or db == 0.0:
            return 0.0
        return num / (da * db)

    def search(self, query: str, top_k: int = 6) -> List[Hit]:
        self._build_bank()
        t0 = time.time()

        if not self._chunk_bank:
            return []

        # Query vorbereiten
        q_vec = self.encoder.encode([query])[0]
        q_toks_list = _tokenize(query)
        q_toks = set(q_toks_list)
        q_toks_content = {t for t in q_toks if t not in STOPWORDS}

        scored: List[Scored] = []
        for i, chunk in enumerate(self._chunk_bank):
            chunk_toks = set(_tokenize(chunk.text))
            chunk_toks_content = {t for t in chunk_toks if t not in STOPWORDS}

            # --- Overlap-Check: mindestens 1 inhaltsreiches Token ---
            overlap_content = len(q_toks_content & chunk_toks_content)
            if overlap_content < MIN_OVERLAP_CONTENT_TOKENS:
                if (time.time() - t0) > self.time_limit_sec:
                    break
                continue

            # semantisch
            s_sem = self._cos(q_vec, self._vecs[i]) if i < len(self._vecs) else 0.0

            # lexical (idf-Summe über Query-Tokens, die im Chunk vorkommen)
            bow = self._bow[i] if i < len(self._bow) else {}
            s_lex = sum(self._idf.get(tok, 0.0) for tok in q_toks if tok in bow)

            # sanfter Overlap-Score (gesamt, inkl. Stopwörter – aber klein gewichtet)
            overlap_total = len(q_toks & chunk_toks)
            s_ovl = overlap_total / max(3, len(q_toks_list))  # 0..~0.66

            # Pfad-Boost
            p = str(chunk.path).lower()
            path_boost = 1.0
            for key, mult in PATH_BOOSTS.items():
                if key in p:
                    path_boost *= mult

            # Endscore
            score = (0.60 * s_sem + 0.30 * s_lex + 0.10 * s_ovl) * path_boost
            scored.append(Scored(chunk, score))

            if (time.time() - t0) > self.time_limit_sec:
                break

        # Sortieren und entduplizieren
        scored.sort(key=lambda s: s.score, reverse=True)
        hits: List[Hit] = []
        seen = set()
        for s in scored:
            key = (str(s.chunk.path), s.chunk.page, s.chunk.text[:80])
            if key in seen:
                continue
            seen.add(key)
            hits.append(Hit(path=s.chunk.path, page=s.chunk.page, snippet=s.chunk.text.strip()))
            if len(hits) >= top_k:
                break

        return hits
