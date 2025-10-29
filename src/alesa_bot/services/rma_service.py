from __future__ import annotations

from typing import Dict, Any, Optional

from src.alesa_bot.services.rma_repo import RMARepo, RMARecord
from src.alesa_bot.services.emailer import send_mail


class RMAService:
    """Handles persistence + customer acknowledgement for RMAs."""

    def __init__(self, repo: RMARepo) -> None:
        self.repo = repo

    def persist_and_notify(self, payload: Dict[str, Any]) -> Optional[str]:
        try:
            rec = RMARecord(
                id=self.repo.new_id(),
                created_at=self.repo.now_iso(),
                updated_at=self.repo.now_iso(),
                status="eingegangen",
                customer=payload.get("customer", {}),
                items=payload.get("items", []),
                reference=payload.get("reference", {}),
                issue=payload.get("issue", {}),
                evidence=payload.get("evidence", {}),
                preferred_action=payload.get("preferred_action", "repair"),
                escalated=False,
            )
            self.repo.add(rec)

            # Best effort: send email to customer
            to = (rec.customer.get("email") or "").strip()
            if to:
                text = (
                    f"Eingang Ihrer Reklamation (RMA)\n\nRMA-ID: {rec.id}\nStatus: {rec.status}\n"
                )
                send_mail(to=to, subject=f"Ihre RMA {rec.id}", text=text)
            return rec.id
        except Exception:
            return None

