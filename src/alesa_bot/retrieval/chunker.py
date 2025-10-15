# src/alesa_bot/retrieval/chunker.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

@dataclass
class Chunk:
    path: Path
    page: int | None
    text: str

def sliding_window(text: str, size: int = 800, overlap: int = 200) -> List[str]:
    if size <= overlap: overlap = max(0, size // 4)
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        if i + size >= len(text): break
        i += (size - overlap)
    return chunks

def explode_pages(path: Path, pages: list[tuple[int|None, str]],
                  size: int = 800, overlap: int = 200) -> List[Chunk]:
    out: List[Chunk] = []
    for page_no, content in pages:
        for c in sliding_window(content, size=size, overlap=overlap):
            out.append(Chunk(path=path, page=page_no, text=c))
    return out
