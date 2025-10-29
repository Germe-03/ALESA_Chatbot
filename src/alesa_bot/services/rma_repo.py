from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RMARecord:
    id: str
    created_at: str
    updated_at: str
    status: str                 # eingegangen | in_pruefung | wartet_auf_kunde | abgeschlossen
    customer: Dict[str, Any]
    items: List[Dict[str, Any]]
    reference: Dict[str, Any]   # auftragsnr, rechnungsnr, kaufdatum
    issue: Dict[str, Any]       # kategorie, beschreibung
    evidence: Dict[str, Any]    # links, hinweise
    preferred_action: str       # repair | replace | refund | return
    escalated: bool = False


class RMARepo:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "rmas.jsonl"
        if not self.path.exists():
            self.path.touch()

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    @staticmethod
    def new_id() -> str:
        return "RMA-" + datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

    def add(self, rec: RMARecord) -> None:
        line = json.dumps(asdict(rec), ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def list(self, limit: int = 200) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not self.path.exists():
            return rows
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

    def get(self, rma_id: str) -> Optional[Dict[str, Any]]:
        rma_id = (rma_id or "").strip()
        if not rma_id:
            return None
        for row in self.list(limit=10000):
            if row.get("id") == rma_id:
                return row
        return None

    def update_status(self, rma_id: str, status: str, escalated: Optional[bool] = None) -> bool:
        # naive: rewrite file
        rows = self.list(limit=10000)
        changed = False
        for r in rows:
            if r.get("id") == rma_id:
                r["status"] = status
                r["updated_at"] = self.now_iso()
                if escalated is not None:
                    r["escalated"] = bool(escalated)
                changed = True
                break
        if changed:
            with open(self.path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return changed

    # Simple SLAs
    def sla_info(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        # Acknowledge within 24h, resolve target 7 days
        try:
            created = datetime.fromisoformat(rec["created_at"].replace("Z", "+00:00"))
        except Exception:
            created = datetime.utcnow()
        ack_due = created + timedelta(hours=24)
        res_due = created + timedelta(days=7)
        now = datetime.utcnow()
        return {
            "ack_due": ack_due.isoformat() + "Z",
            "resolve_due": res_due.isoformat() + "Z",
            "ack_overdue": now > ack_due,
            "resolve_overdue": now > res_due,
        }

