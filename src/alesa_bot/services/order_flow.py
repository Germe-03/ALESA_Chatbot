# src/alesa_bot/services/order_flow.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import re

# ==== Trigger-Phrasen (Intent von außen) ====
BESTELL_TRIGGER = (
    "bestellen",
    "ich würde gerne bestellen",
    "kann ich gleich bestellen",
    "kann ich direkt bestellen",
    "möchte bestellen",
    "bitte bestellen",
)

# ==== Datenmodelle ====
@dataclass
class CustomerInfo:
    kundennummer: Optional[str] = None               # wahlpflicht
    typ: Optional[str] = None                        # "Privat" | "Firma" (Pflicht, außer wenn Kundennummer)
    name: Optional[str] = None                       # Pflicht, außer wenn Kundennummer
    strasse_nr: Optional[str] = None                 # Pflicht, außer wenn Kundennummer
    plz_ort: Optional[str] = None                    # Pflicht, außer wenn Kundennummer
    bearbeiter: Optional[str] = None                 # Pflicht, wenn Firma
    email: Optional[str] = None                      # Pflicht (immer)

    def need_basic_fields(self) -> bool:
        """Ob Grunddaten (typ, name, strasse_nr, plz_ort, bearbeiter) erfasst werden müssen."""
        return not bool(self.kundennummer and self.kundennummer.strip())

    def required_missing(self) -> List[str]:
        missing: List[str] = []
        # E-Mail immer Pflicht
        if not self.email:
            missing.append("email")
        # Wenn Kundennummer vorhanden, keine weiteren Pflichtfelder außer E-Mail
        if not self.need_basic_fields():
            return missing
        # Ohne Kundennummer: Pflichtfelder
        if not self.typ:
            missing.append("typ")
        if not self.name:
            missing.append("name")
        if not self.strasse_nr:
            missing.append("strasse_nr")
        if not self.plz_ort:
            missing.append("plz_ort")
        if self.typ == "Firma" and not self.bearbeiter:
            missing.append("bearbeiter")
        return missing

@dataclass
class OrderItem:
    artikelnummer: Optional[str] = None  # Pflicht
    menge: Optional[str] = None          # Pflicht

    def is_complete(self) -> bool:
        return bool(self.artikelnummer and self.menge)

@dataclass
class OrderState:
    phase: str = "idle"                  # idle | customer | items | comment | confirm
    customer: CustomerInfo = field(default_factory=CustomerInfo)
    items: List[OrderItem] = field(default_factory=list)
    current_idx: int = -1
    ask_field: Optional[str] = None      # aktueller Slotname
    kommentar: Optional[str] = None      # optional
    summary_cache: str = ""

# ==== Hilfen ====
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _normalize_typ(value: str) -> Optional[str]:
    v = value.strip().lower()
    if v in {"privat", "p"}:
        return "Privat"
    if v in {"firma", "unternehmen", "u"}:
        return "Firma"
    return None

# ==== Bestell-Flow ====
class OrderFlow:
    """
    Bestell-Assistent:
      1) Kundendaten:   Kundennummer (wahlpflicht) →
                        Wenn leer: Typ (Privat/Firma), Name, Straße/Nr, PLZ/Ort (+ Bearbeiter bei Firma)
                        E-Mail immer Pflicht.
      2) Artikel:       Beliebig viele Positionen, je Artikel: Artikelnummer (Pflicht), Menge (Pflicht).
                        Befehle: 'neuer artikel', 'fertig'
      3) Kommentar:     Wahl (Enter zum Überspringen)
      4) Zusammenfassung & Bestätigung: 'ja'/'nein'
    """
    def __init__(self) -> None:
        self.state = OrderState()

    # --- Public API ---
    def is_active(self) -> bool:
        return self.state.phase != "idle"

    def should_trigger(self, text: str) -> bool:
        t = text.strip().lower()
        return any(k in t for k in BESTELL_TRIGGER)

    def start(self) -> str:
        self.state = OrderState(phase="customer", customer=CustomerInfo(), items=[], current_idx=-1,
                                ask_field=None, kommentar=None, summary_cache="")
        # Erste Frage: Kundennummer (wahlpflicht, Enter überspringt)
        self.state.ask_field = "kundennummer"
        return ("Alles klar – ich starte den Bestell-Assistenten.\n"
                "Zuerst benötige ich deine Kundendaten.\n\n"
                "Kundennummer? (optional – mit Enter überspringen)")

    def handle(self, user_text: str) -> str:
        # Hinweis: Leere Eingaben (Enter) sind erlaubt → wichtig für optionale Felder!
        t = (user_text or "")
        low = t.strip().lower()

        if self.state.phase == "idle":
            return "Wenn du eine Bestellung starten möchtest, schreibe z. B.: „Ich würde gerne bestellen.“"

        if self.state.phase == "customer":
            return self._handle_customer_phase(t)

        if self.state.phase == "items":
            # Steuerbefehle:
            if low == "neuer artikel":
                if self._current_item() and not self._current_item().is_complete():
                    return "Bitte schließe den aktuellen Artikel zuerst ab (Artikelnummer und Menge). " + self._ask_next_item_slot()
                self._add_new_item()
                return "Neuer Artikel – alles klar.\n" + self._ask_next_item_slot()

            if low == "fertig":
                # Prüfen: mind. 1 vollständiger Artikel?
                if not any(it.is_complete() for it in self.state.items):
                    return "Es wurde noch kein vollständiger Artikel erfasst. Bitte mindestens Artikelnummer und Menge angeben."
                # Weiter zur Kommentar-Phase
                self.state.phase = "comment"
                self.state.ask_field = "kommentar"
                return "Kommentar zur Bestellung? (optional – mit Enter überspringen)"

            # Slot-Filling Artikel
            return self._fill_item_slot_and_ask_next(t)

        if self.state.phase == "comment":
            # Kommentar (optional – Enter = überspringen)
            if t.strip():
                self.state.kommentar = t.strip()
            else:
                self.state.kommentar = ""
            # Weiter: Zusammenfassung/Bestätigung
            self.state.phase = "confirm"
            self.state.ask_field = None
            self.state.summary_cache = self._build_summary()
            return self.state.summary_cache + "\n\nBitte bestätigen: **ja** / **nein**"

        if self.state.phase == "confirm":
            if low == "ja":
                self.state.phase = "idle"
                return "✅ Besten Dank für Ihre Bestellung, kann ich Ihnen sonst noch weiterhelfen?"
            if low == "nein":
                # Zurück in Artikel-Phase, um zu ändern / neue Artikel
                self.state.phase = "items"
                # Wenn noch kein aktiver Artikel offen ist, neuen beginnen
                if self.state.current_idx < 0 or self._current_item().is_complete():
                    self._add_new_item()
                return "Kein Problem. Was möchten Sie anpassen? (Tipp: 'neuer artikel' oder Felder überschreiben)\n" + self._ask_next_item_slot()
            return "Bitte antworte mit **ja** oder **nein**."

        return "Unbekannter Zustand. Starte bei Bedarf neu mit „Ich würde gerne bestellen.“"

    # --- Customer Phase ---
    def _handle_customer_phase(self, user_text: str) -> str:
        cs = self.state.customer
        field = self.state.ask_field

        # 1) Kundennummer (wahlpflicht)
        if field == "kundennummer":
            # Leere Eingabe → überspringen
            val = user_text.strip()
            if val:
                cs.kundennummer = val
            # Nächster Pflichtblock
            if cs.need_basic_fields():
                self.state.ask_field = "typ"
                return "Privat oder Firma? (Pflicht – antworte mit 'Privat' oder 'Firma')"
            else:
                # Kundennummer vorhanden → direkt E-Mail (immer Pflicht)
                self.state.ask_field = "email"
                return "E-Mail für Bestätigung? (Pflicht)"

        # 2) Typ (Pflicht, nur wenn keine Kundennummer)
        if field == "typ":
            typ = _normalize_typ(user_text)
            if not typ:
                return "Bitte 'Privat' oder 'Firma' angeben."
            cs.typ = typ
            self.state.ask_field = "name"
            return "Name? (Pflicht)"

        # 3) Name (Pflicht)
        if field == "name":
            if not user_text.strip():
                return "Name ist erforderlich."
            cs.name = user_text.strip()
            self.state.ask_field = "strasse_nr"
            return "Strasse / Nr.? (Pflicht)"

        # 4) Strasse / Nr. (Pflicht)
        if field == "strasse_nr":
            if not user_text.strip():
                return "Strasse / Nr. ist erforderlich."
            cs.strasse_nr = user_text.strip()
            self.state.ask_field = "plz_ort"
            return "PLZ / Ort? (Pflicht)"

        # 5) PLZ / Ort (Pflicht)
        if field == "plz_ort":
            if not user_text.strip():
                return "PLZ / Ort ist erforderlich."
            cs.plz_ort = user_text.strip()
            # Wenn Firma → Bearbeiter, sonst E-Mail
            if cs.typ == "Firma":
                self.state.ask_field = "bearbeiter"
                return "Bearbeiter/in? (Pflicht)"
            else:
                self.state.ask_field = "email"
                return "E-Mail für Bestätigung? (Pflicht)"

        # 6) Bearbeiter (Pflicht, wenn Firma)
        if field == "bearbeiter":
            if not user_text.strip():
                return "Bearbeiter/in ist erforderlich."
            cs.bearbeiter = user_text.strip()
            self.state.ask_field = "email"
            return "E-Mail für Bestätigung? (Pflicht)"

        # 7) E-Mail (Pflicht, immer)
        if field == "email":
            email = user_text.strip()
            if not email or not _EMAIL_RE.match(email):
                return "Bitte eine gültige E-Mail-Adresse angeben."
            cs.email = email

            # Weiter zu Artikel-Phase
            self.state.phase = "items"
            self.state.ask_field = None
            self._add_new_item()
            return ("Danke. Nun zu den Artikeln.\n"
                    + self._ask_next_item_slot())

        return "Bitte gib die angefragte Information an."

    # --- Items Phase ---
    def _add_new_item(self) -> None:
        self.state.items.append(OrderItem())
        self.state.current_idx = len(self.state.items) - 1
        self.state.ask_field = "artikelnummer"

    def _current_item(self) -> Optional[OrderItem]:
        if self.state.current_idx < 0 or self.state.current_idx >= len(self.state.items):
            return None
        return self.state.items[self.state.current_idx]

    def _ask_next_item_slot(self) -> str:
        it = self._current_item()
        if it is None:
            self._add_new_item()
            it = self._current_item()
        if not it.artikelnummer:
            self.state.ask_field = "artikelnummer"
            return "Artikelnummer? (Pflicht)"
        if not it.menge:
            self.state.ask_field = "menge"
            return "Menge? (Pflicht)"
        self.state.ask_field = None
        return ("Dieser Artikel ist vollständig. Du kannst **'neuer artikel'** sagen "
                "oder **'fertig'**, um fortzufahren.")

    def _fill_item_slot_and_ask_next(self, user_value: str) -> str:
        it = self._current_item()
        if it is None:
            self._add_new_item()
            it = self._current_item()

        field_name = self.state.ask_field

        # Artikelnummer (Pflicht)
        if field_name == "artikelnummer":
            if not user_value.strip():
                return "Artikelnummer ist erforderlich."
            it.artikelnummer = user_value.strip()
            return self._ask_next_item_slot()

        # Menge (Pflicht)
        if field_name == "menge":
            if not user_value.strip():
                return "Menge ist erforderlich."
            it.menge = user_value.strip()
            return self._ask_next_item_slot()

        # Kein Slot offen → Hinweis
        return ("Wenn du einen weiteren Artikel anlegen willst, schreibe **'neuer artikel'**. "
                "Oder **'fertig'** zum Abschluss der Artikelerfassung.")

    # --- Summary ---
    def _build_summary(self) -> str:
        cs = self.state.customer
        lines: List[str] = ["🧾 Zusammenfassung Ihrer Bestellanfrage:"]

        # Kunde
        if cs.kundennummer:
            lines.append(f"Kundennummer: {cs.kundennummer}")
        else:
            lines.append(f"Typ: {cs.typ}")
            lines.append(f"Name: {cs.name}")
            lines.append(f"Strasse / Nr.: {cs.strasse_nr}")
            lines.append(f"PLZ / Ort: {cs.plz_ort}")
            if cs.typ == "Firma":
                lines.append(f"Bearbeiter/in: {cs.bearbeiter}")
        lines.append(f"E-Mail: {cs.email}")

        # Artikel
        lines.append("\nPositionen:")
        pos_idx = 1
        for it in self.state.items:
            if not it.is_complete():
                continue
            lines.append(f"  {pos_idx}) Artikelnummer: {it.artikelnummer} – Menge: {it.menge}")
            pos_idx += 1

        # Kommentar (optional)
        if self.state.kommentar:
            lines.append(f"\nKommentar: {self.state.kommentar}")

        return "\n".join(lines)
