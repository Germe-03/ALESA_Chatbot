from __future__ import annotations

from typing import Any, Dict, Optional

from src.alesa_bot.services.emailer import send_order_confirmation
from src.alesa_bot.services.order_repo import OrderRecord, OrderRepo


class OrderService:
    """Handles persistence + customer email for orders."""

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
            customer_name = payload.get("customer", {}).get("name", "")
            if to:
                send_order_confirmation(
                    to=to,
                    order_id=rec.id,
                    items=payload.get("items", []),
                    customer_name=customer_name,
                    comment=payload.get("comment"),
                )

            return rec.id
        except Exception:
            return None
