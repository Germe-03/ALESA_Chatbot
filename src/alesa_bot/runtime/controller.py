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


class AppController:
    def __init__(
        self,
        qa_service: QAService,
        order_flow: OrderFlow,
        pre_order_gate: PreOrderGate,
        system_banner: str = "",
    ) -> None:
        self.qa_service = qa_service
        self.order_flow = order_flow
        self.pre_gate = pre_order_gate
        self.system_banner = system_banner

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

        # ------------------------
        # A) Aktiver Bestell-Flow
        # ------------------------
        if self.order_flow.is_active():
            reply = self.order_flow.handle(q)
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
                responses.append("Bitte antworte mit **bestellen** oder **beratung**.")
                return responses

        # ------------------------
        # C) Gate bei Kaufintention
        # ------------------------
        if not self.pre_gate.suppress_next_gate and self.pre_gate.should_prompt_gate(q):
            prompt = self.pre_gate.start(q)
            responses.append(prompt)
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
        return responses
