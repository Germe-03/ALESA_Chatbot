from __future__ import annotations
"""
Intent- und Entscheidungslogik (Pre-Order-Gate) für ALESA.

Dieses Modul kapselt:
- Erkennung, ob eine Eingabe eine Frage ist
- Erkennung starker Kaufabsicht (Purchase Intent)
- Zustandsautomat PreOrderGate, der zwischen "Bestellen" und "Beratung" verzweigt

Kein IO in diesem Modul – reine Logik, damit gut testbar und UI-agnostisch.
"""

from typing import Optional
import re as _re


# ---------- Interrogativ-Wörter / Frageerkennung ----------

_INTERROGATIVES = (
    "was", "welche", "welcher", "welches", "wie", "wo", "wann",
    "wer", "wieso", "weshalb", "warum", "kann", "darf", "soll",
)

def is_question(text: str) -> bool:
    """Heuristik: enthält '?' oder beginnt mit Interrogativ."""
    t = (text or "").strip()
    if not t:
        return False
    if "?" in t:
        return True
    start = t.split(maxsplit=1)[0].lower()
    return start in _INTERROGATIVES


# ---------- Kaufintention / Strong Purchase Intent ----------

_PURCHASE_REGEXES = [
    _re.compile(r"\bich\s+(?:möchte|will|würde(?:\s+gern(?:e)?)?)\b.*\bbestellen\b", _re.IGNORECASE),
    _re.compile(r"\bkann\s+(?:ich|man)\b.*\bbestellen\b", _re.IGNORECASE),
    _re.compile(r"\bbitte\b.*\bbestellen\b", _re.IGNORECASE),
    _re.compile(r"\b(?:ich\s+)?bestelle\b.*", _re.IGNORECASE),
    _re.compile(r"\bmöchte\b.*\bbestellen\b", _re.IGNORECASE),
]

def is_strong_purchase_intent(text: str) -> bool:
    """Positive Kaufabsicht, aber keine Frage."""
    t = (text or "").strip()
    if not t:
        return False
    if is_question(t):
        return False
    return any(rx.search(t) for rx in _PURCHASE_REGEXES)


# ---------- PreOrderGate ----------

class PreOrderGate:
    """
    Zustandsautomat, der – bei starker Kaufabsicht – den Nutzer fragt:
      "Direkt bestellen" oder "Beratung/Produktempfehlung"?
    und entsprechend in den Bestell-Flow bzw. QA verzweigt.

    Attribute:
      await_choice:     True, wenn eine Antwort "bestellen" oder "beratung" erwartet wird
      product_hint:     evtl. erkannter Produktbegriff (z. B. "Nutex") aus der Nutzerfrage
      cached_user_query:Originale Nutzerfrage, falls erst Beratung gewählt wird
      mode:             "neutral" | "advice" (nur informativ)
      suppress_next_gate:Einmaliges Unterdrücken der Gate-Abfrage (z. B. nach Beratung)
    """
    def __init__(self) -> None:
        self.await_choice: bool = False
        self.product_hint: str = ""
        self.cached_user_query: str = ""
        self.mode: str = "neutral"          # "neutral" | "advice"
        self.suppress_next_gate: bool = False

    # ---- interne Helfer ----

    @staticmethod
    def _extract_product_hint(text: str) -> str:
        low = (text or "").lower()
        if "nutex" in low:
            return "Nutex"
        return ""

    # ---- öffentliche API ----

    def start(self, user_text: str) -> str:
        """
        Aktiviert das Gate und fordert den Nutzer zur Entscheidung auf.
        Gibt die Prompt-Nachricht zurück.
        """
        self.await_choice = True
        self.cached_user_query = user_text
        self.product_hint = self._extract_product_hint(user_text)
        prod = f" **{self.product_hint}**" if self.product_hint else ""
        return (
            f"Möchten Sie{prod} **direkt bestellen** oder zuerst eine **Beratung/Produktempfehlung**?\n"
            "Bitte antworten Sie mit **bestellen** oder **beratung**."
        )

    def active(self) -> bool:
        """True, wenn das Gate auf eine Entscheidung wartet."""
        return self.await_choice

    def handle_choice(self, user_text: str) -> str:
        """
        Verarbeitet die Entscheidung des Nutzers.
        Rückgaben:
          - "go_order": Bestell-Flow starten
          - "go_qa":    Beratung/QA starten
          - "repeat":   unklare Eingabe – erneut fragen
        """
        low = (user_text or "").strip().lower()
        if "bestellen" in low:
            self.await_choice = False
            self.mode = "neutral"
            return "go_order"
        if "beratung" in low or "empfehlung" in low:
            self.await_choice = False
            self.mode = "advice"
            self.suppress_next_gate = True
            return "go_qa"
        return "repeat"

    def should_prompt_gate(self, user_text: str) -> bool:
        """
        Entscheidet, ob das Gate überhaupt gefragt werden soll.
        Wir fragen NICHT bei Fragen, nur bei starker Kaufabsicht.
        """
        if is_question(user_text):
            return False
        return is_strong_purchase_intent(user_text)
