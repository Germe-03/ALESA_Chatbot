# ===================== FILE: src/alesa_bot/adapters/file_io.py =====================
from __future__ import annotations
from pathlib import Path
from typing import Iterable


def iter_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root or not root.exists():
            continue
        yield from (p for p in root.rglob('*') if p.is_file())