# ===================== FILE: src/alesa_bot/core/types.py =====================
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, List, Optional


@dataclass
class Hit:
    path: Path
    page: Optional[int]
    snippet: str


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 6) -> List[Hit]: ...


class LLM(Protocol):
    def start(self) -> None: ...
    def generate(self, prompt: str) -> str: ...