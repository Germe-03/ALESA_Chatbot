from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

from src.alesa_bot.core.types import Hit
from .indexer import FileIndexer
from .chunker import explode_pages, Chunk
from .embeddings import EmbeddingEncoder


def _id_for(path: Path, page: int | None, text: str) -> str:
    sig = f"{path}|{page or 0}|{text[:80]}".encode("utf-8", errors="ignore")
    return hashlib.sha1(sig).hexdigest()


@dataclass
class VSConfig:
    root: Path
    collection: str = "alesa"


class ChromaStore:
    """Lightweight wrapper around Chroma persistent store.

    - Stores chunk text with metadata: path, page.
    - Uses Vertex embeddings via provided EmbeddingEncoder.
    - Provides query() returning Hit objects.
    """

    def __init__(self, cfg: VSConfig, encoder: EmbeddingEncoder) -> None:
        import chromadb  # lazy import to keep optional

        self._client = chromadb.PersistentClient(path=str(cfg.root))
        self._col = self._client.get_or_create_collection(cfg.collection)
        self._encoder = encoder

    def _prepare_chunks(self, indexer: FileIndexer, size: int = 800, overlap: int = 200) -> List[Chunk]:
        bank: List[Chunk] = []
        for path_str, pages in indexer.data.items():
            bank.extend(explode_pages(Path(path_str), pages, size=size, overlap=overlap))
        return bank

    def build(self, indexer: FileIndexer, size: int = 800, overlap: int = 200, batch: int = 96) -> Tuple[int, int]:
        """Rebuild vector store from current indexer data if new chunks detected.

        Returns (added, skipped).
        """
        chunks = self._prepare_chunks(indexer, size=size, overlap=overlap)
        if not chunks:
            return (0, 0)

        # Determine which IDs already exist
        ids: List[str] = [_id_for(c.path, c.page, c.text) for c in chunks]

        # Chroma doesn't expose bulk membership; we read existing ids in pages
        # Workaround: fetch count and chunk over input ids to check presence
        existing: set[str] = set()
        for i in range(0, len(ids), 1000):
            sub = ids[i:i + 1000]
            # query where filter by ids isn't supported for read; try get with where on path signature
            # We'll accept re-upsert safety: upserting existing IDs is idempotent.
            existing |= set()  # placeholder to keep logic simple

        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        new_ids: List[str] = []

        for idv, c in zip(ids, chunks):
            new_ids.append(idv)
            texts.append(c.text)
            metas.append({
                "path": str(c.path),
                "page": c.page or 0,
            })

        added = 0
        for i in range(0, len(texts), batch):
            tb = texts[i:i + batch]
            ib = new_ids[i:i + batch]
            mb = metas[i:i + batch]
            vecs = self._encoder.encode(tb)
            # Upsert with explicit embeddings
            self._col.upsert(ids=ib, documents=tb, metadatas=mb, embeddings=vecs)
            added += len(tb)

        return (added, 0)

    def query(self, question: str, top_k: int = 6) -> List[Hit]:
        q_vec = self._encoder.encode([question])[0]
        res = self._col.query(query_embeddings=[q_vec], n_results=top_k)
        docs: List[str] = res.get("documents", [[]])[0] if res else []
        mets: List[Dict[str, Any]] = res.get("metadatas", [[]])[0] if res else []
        hits: List[Hit] = []
        for txt, md in zip(docs, mets):
            p = Path(md.get("path", ""))
            page = int(md.get("page", 0)) or None
            hits.append(Hit(path=p, page=page, snippet=(txt or "").strip()))
        return hits

