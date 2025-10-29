from __future__ import annotations

from typing import Dict, Any, Optional

from src.alesa_bot.services.order_repo import OrderRepo, OrderRecord
from src.alesa_bot.services.emailer import send_mail


class OrderService:
    """Handles persistence + customer email for orders.

    Expects payload as produced by OrderFlow in AppController._order_payload().
    """

    def __init__(self, repo: OrderRepo) -> None:
        self.repo = repo

    def persist_and_notify(self, payload: Dict[str, Any]) -> Optional[str]:
        try:
            rec = OrderRecord(
                id=self.repo.new_id(),
                created_at=self.repo.now_iso(),
                customer=payload.get("customer", {}),
                items=payload.get("items", []),
                comment=payload.get("comment"),
            )
            self.repo.add(rec)

            # Best effort: send confirmation email
            to = (payload.get("customer", {}).get("email") or "").strip()
            if to:
                lines = ["Bestellbestaetigung", "", f"Bestell-ID: {rec.id}"]
                for it in payload.get("items", []):
                    lines.append(f"- {it.get('artikelnummer','?')} x {it.get('menge','?')}")
                if payload.get("comment"):
                    lines += ["", f"Kommentar: {payload['comment']}"]
                send_mail(to=to, subject="Ihre ALESA Bestellbestaetigung", text="\n".join(lines))

            return rec.id
        except Exception:
            return None

