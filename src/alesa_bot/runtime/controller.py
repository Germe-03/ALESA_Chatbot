from __future__ import annotations
"""
AppController – Dialog-Orchestrierung zwischen OrderFlow, PreOrderGate und QAService.

Diese Schicht bündelt:
- Intentsteuerung (Gate)
- Bestellabläufe (OrderFlow)
- Wissensabfragen (QAService)

Der Controller kennt keine Ein-/Ausgabe. Er liefert nur Text-Antworten
(zur Anzeige durch die UI-Schicht).
"""

from typing import List
from src.alesa_bot.services.order_flow import OrderFlow
from src.alesa_bot.services.qa_service import QAService
from src.alesa_bot.assistant.preorder_gate import PreOrderGate
from src.alesa_bot.services.order_repo import OrderRepo
from src.alesa_bot.services.order_service import OrderService
from src.alesa_bot.services.rma_flow import RMAFlow
from src.alesa_bot.services.rma_repo import RMARepo
from src.alesa_bot.services.rma_service import RMAService
from datetime import datetime


class AppController:
    def __init__(
        self,
        qa_service: QAService,
        order_flow: OrderFlow,
        pre_order_gate: PreOrderGate,
        system_banner: str = "",
        orders_repo: OrderRepo | None = None,
        order_service: OrderService | None = None,
        rma_flow: RMAFlow | None = None,
        rma_repo: RMARepo | None = None,
        rma_service: RMAService | None = None,
    ) -> None:
        self.qa_service = qa_service
        self.order_flow = order_flow
        self.pre_gate = pre_order_gate
        self.system_banner = system_banner
        self.orders_repo = orders_repo
        self.order_service = order_service
        self.rma_flow = rma_flow or RMAFlow()
        self.rma_repo = rma_repo
        self.rma_service = rma_service
        # bind persistence callback
        try:
            self.rma_flow._on_submit = self.persist_rma  # type: ignore[attr-defined]
        except Exception:
            pass
        self._await_order_confirm = False

    # ----------------------------------------------------
    # Öffentliche Schnittstelle (vom Runner genutzt)
    # ----------------------------------------------------

    def on_start(self) -> str | None:
        """Optionaler Initialtext (hier leer, da Banner bereits UI-seitig gezeigt wird)."""
        return None

    def handle(self, user_text: str) -> List[str]:
        """
        Zentraler Dispatcher: nimmt eine Nutzereingabe entgegen
        und gibt eine Liste von Bot-Antworten zurück.
        """
        responses: List[str] = []

        q = (user_text or "").strip()
        low = q.lower()

        # E-1) Wenn exakt "bestellen" eingegeben wird: Bestellvorgang starten
        if not self.order_flow.is_active() and low == "bestellen":
            responses.append(self.order_flow.start())
            return responses

        # E0) Await order confirmation after product answer
        if getattr(self, "_await_order_confirm", False) and not self.order_flow.is_active():
            if low in {"ja", "j", "yes", "y", "bestellen"}:
                self._await_order_confirm = False
                responses.append(self.order_flow.start())
                return responses
            if low in {"nein", "n", "no"}:
                self._await_order_confirm = False
                responses.append("Alles klar. Sagen Sie Bescheid, wenn Sie bestellen möchten.")
                return responses
            responses.append("Bitte antworten Sie mit **ja** (bestellen) oder **nein**.")
            return responses

        # ------------------------
        # A) Aktiver Bestell- oder RMA-Flow
        # ------------------------
        if self.order_flow.is_active():
            prev_phase = self.order_flow.state.phase
            reply = self.order_flow.handle(q)
            # Persist order when confirming from 'confirm' to 'idle' with a positive answer
            try:
                if self.orders_repo is not None and prev_phase == "confirm" and (q or "").strip().lower().startswith("ja") and not self.order_flow.is_active():
                    payload = self._order_payload()
                    if payload and self.order_service is not None:
                        self.order_service.persist_and_notify(payload)
            except Exception:
                pass
            responses.append(reply)
            return responses

        if self.rma_flow.is_active():
            reply = self.rma_flow.handle(q)
            responses.append(reply)
            return responses

        # ------------------------
        # B) Aktives Pre-Order-Gate
        # ------------------------
        if self.pre_gate.active():
            action = self.pre_gate.handle_choice(q)
            if action == "go_order":
                responses.append(self.order_flow.start())
                return responses
            elif action == "go_qa":
                # ursprüngliche Nutzerfrage aus Cache übernehmen
                cached = self.pre_gate.cached_user_query or q
                self.pre_gate.cached_user_query = ""
                # kein return → weiter unten QA
                q = cached
            elif action == "repeat":
                responses.append("Bitte antworten Sie mit **bestellen** oder **beratung**.")
                return responses

        # ------------------------
        # C) Gate bei Kaufintention
        # ------------------------
        # C) Gate nur, wenn nicht lediglich das Wort 'bestellen' im Satz vorkommt
        if ("bestellen" not in low) and (not self.pre_gate.suppress_next_gate) and self.pre_gate.should_prompt_gate(q):
            prompt = self.pre_gate.start(q)
            responses.append(prompt)
            return responses

        # C2) Reklamation: bei exakt 'reklamation' oder 'retoure' etc. starten
        if not self.rma_flow.is_active():
            simple_triggers = {"reklamation", "retoure", "umtausch", "garantie"}
            if low in simple_triggers:
                responses.append(self.rma_flow.start())
                return responses

        # C3) RMA-Statusabfrage: 'status <RMA-ID>' oder 'rma <id>'
        if self.rma_repo is not None:
            parts = low.split()
            if parts and parts[0] in {"status", "rma"} and len(parts) > 1:
                rid = user_text.split(maxsplit=1)[1].strip()
                rec = self.rma_repo.get(rid)
                if rec:
                    sla = self.rma_repo.sla_info(rec)
                    responses.append(
                        f"Status {rec['id']}: {rec.get('status','-')}\n"
                        f"Erstellt: {rec.get('created_at','-')} | Aktualisiert: {rec.get('updated_at','-')}\n"
                        f"SLA: ack bis {sla['ack_due']} ({'ueberfaellig' if sla['ack_overdue'] else 'in Frist'}), "
                        f"loesen bis {sla['resolve_due']} ({'ueberfaellig' if sla['resolve_overdue'] else 'in Frist'})"
                    )
                else:
                    responses.append("Keine Reklamation mit dieser ID gefunden.")
                return responses

        # Einmalige Gate-Unterdrückung zurücksetzen
        if self.pre_gate.suppress_next_gate:
            self.pre_gate.suppress_next_gate = False

        # ------------------------
        # D) Standard-QA
        # ------------------------
        if not q:
            # Leere Eingabe: kein Output
            return []

        try:
            answer, cites = self.qa_service.ask(q)
        except ValueError as ve:
            responses.append(str(ve))
            return responses
        except Exception as e:
            responses.append(f"⚠️  Fehler bei der Verarbeitung: {e}")
            return responses

        # QA-Ergebnis zusammenbauen
        responses.append(answer)
        if cites:
            responses.append("Quellen:\n" + "\n".join(cites))
        # Mark that we await order confirmation if the last QA was a product hit
        try:
            if getattr(self.qa_service, "last_product", None) is not None and not self.order_flow.is_active():
                self._await_order_confirm = True
        except Exception:
            pass
        return responses

    # -------------- helpers --------------
    def _order_payload(self):
        try:
            cs = self.order_flow.state.customer
            items = [
                {"artikelnummer": it.artikelnummer, "menge": it.menge}
                for it in self.order_flow.state.items if it.is_complete()
            ]
            return {
                "customer": {
                    "kundennummer": cs.kundennummer,
                    "typ": cs.typ,
                    "name": cs.name,
                    "strasse_nr": cs.strasse_nr,
                    "plz_ort": cs.plz_ort,
                    "bearbeiter": cs.bearbeiter,
                    "email": cs.email,
                },
                "items": items,
                "comment": self.order_flow.state.kommentar,
            }
        except Exception:
            return None

    # -------------- RMA helpers --------------
    def persist_rma(self, payload: dict) -> str | None:
        try:
            if self.rma_service is None:
                return None
            return self.rma_service.persist_and_notify(payload)  # type: ignore[return-value]
        except Exception:
            return None
