from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class OrderRecord:
    id: str               # timestamp-based id
    created_at: str       # ISO 8601
    customer: Dict[str, Any]
    items: List[Dict[str, Any]]
    comment: str | None = None


class OrderRepo:
    """Very simple JSONL-backed repository for orders."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "orders.jsonl"
        if not self.path.exists():
            self.path.touch()

    def add(self, rec: OrderRecord) -> None:
        line = json.dumps(asdict(rec), ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def list(self, limit: int = 200) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows[-limit:]

    @staticmethod
    def new_id() -> str:
        return datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

