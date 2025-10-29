from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
import re


REKLAMATION_TRIGGER = (
    "reklamation", "reklamieren", "garantie", "defekt", "kaputt", "beschaedigt",
    "beschädigt", "rueckgabe", "rückgabe", "retoure", "umtausch", "falsch geliefert",
)


@dataclass
class RmaCustomer:
    email: Optional[str] = None
    kundennummer: Optional[str] = None
    typ: Optional[str] = None  # Privat/Firma
    name: Optional[str] = None


@dataclass
class RmaItem:
    artikelnummer: Optional[str] = None
    menge: Optional[str] = None

    def is_complete(self) -> bool:
        return bool(self.artikelnummer and self.menge)


@dataclass
class RmaState:
    phase: str = "idle"  # idle | customer | reference | items | issue | evidence | preference | confirm
    ask_field: Optional[str] = None
    customer: RmaCustomer = field(default_factory=RmaCustomer)
    auftragsnr: Optional[str] = None
    rechnungsnr: Optional[str] = None
    kaufdatum: Optional[str] = None
    items: List[RmaItem] = field(default_factory=list)
    current_idx: int = -1
    kategorie: Optional[str] = None  # defekt/beschädigt/falsch
    beschreibung: Optional[str] = None
    evidence_links: List[str] = field(default_factory=list)
    preferred_action: Optional[str] = None  # repair | replace | refund | return
    summary_cache: str = ""


class RMAFlow:
    def __init__(self, on_submit: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self.state = RmaState()
        self._on_submit = on_submit

    # --------------- helpers ---------------
    def is_active(self) -> bool:
        return self.state.phase != "idle"

    def should_trigger(self, text: str) -> bool:
        t = (text or "").strip().lower()
        return t in REKLAMATION_TRIGGER

    def _add_new_item(self) -> None:
        self.state.items.append(RmaItem())
        self.state.current_idx = len(self.state.items) - 1
        self.state.ask_field = "artikelnummer"

    def _current_item(self) -> Optional[RmaItem]:
        if self.state.current_idx < 0 or self.state.current_idx >= len(self.state.items):
            return None
        return self.state.items[self.state.current_idx]

    # --------------- flow ---------------
    def start(self) -> str:
        self.state = RmaState(phase="customer")
        self.state.ask_field = "email"
        return (
            "Alles klar – ich eröffne eine Reklamation (RMA).\n"
            "Bitte zuerst deine E‑Mail (Pflicht)."
        )

    def handle(self, user_text: str) -> str:
        t = (user_text or "").strip()
        low = t.lower()

        if self.state.phase == "idle":
            return "Schreibe 'reklamation', um eine RMA zu eröffnen."

        if self.state.phase == "customer":
            return self._handle_customer(t)

        if self.state.phase == "reference":
            return self._handle_reference(t)

        if self.state.phase == "items":
            if low == "neuer artikel":
                if self._current_item() and not self._current_item().is_complete():
                    return "Bitte gib erst Artikelnummer und Menge an."
                self._add_new_item()
                return "Neuer Artikel – bitte Artikelnummer."
            return self._handle_items(t)

        if self.state.phase == "issue":
            return self._handle_issue(t)

        if self.state.phase == "evidence":
            return self._handle_evidence(t)

        if self.state.phase == "preference":
            return self._handle_preference(t)

        if self.state.phase == "confirm":
            if low in {"ja", "j", "yes", "y"}:
                try:
                    if callable(self._on_submit):
                        self._on_submit(self._to_payload())
                except Exception:
                    pass
                self.state.phase = "idle"
                return "Danke – deine Reklamation wurde erfasst. Du erhältst eine Bestätigung per E‑Mail."
            if low in {"nein", "n", "no"}:
                self.state.phase = "items"
                if self.state.current_idx < 0 or self._current_item().is_complete():
                    self._add_new_item()
                return "Kein Problem. Was möchtest du ändern?"
            return "Bitte bestätige mit **ja** oder **nein**."

        return "Unbekannter Zustand. Schreibe 'reklamation', um neu zu starten."

    # ---- phases ----
    def _handle_customer(self, value: str) -> str:
        cs = self.state.customer
        if self.state.ask_field == "email":
            if not value:
                return "E‑Mail ist erforderlich."
            cs.email = value
            self.state.ask_field = "kundennummer"
            return "Kundennummer? (optional – Enter zum Überspringen)"
        if self.state.ask_field == "kundennummer":
            cs.kundennummer = value or None
            self.state.ask_field = "typ"
            return "Privat oder Firma? (optional)"
        if self.state.ask_field == "typ":
            cs.typ = value or None
            self.state.ask_field = "name"
            return "Name? (optional)"
        if self.state.ask_field == "name":
            cs.name = value or None
            # next phase
            self.state.phase = "reference"
            self.state.ask_field = "auftragsnr"
            return "Auftrags- oder Rechnungsnummer? (optional)"
        return "Bitte gib die angefragte Information an."

    def _handle_reference(self, value: str) -> str:
        if self.state.ask_field == "auftragsnr":
            self.state.auftragsnr = value or None
            self.state.ask_field = "rechnungsnr"
            return "Rechnungsnummer? (optional)"
        if self.state.ask_field == "rechnungsnr":
            self.state.rechnungsnr = value or None
            self.state.ask_field = "kaufdatum"
            return "Kauf-/Lieferdatum? (optional)"
        if self.state.ask_field == "kaufdatum":
            self.state.kaufdatum = value or None
            # next
            self.state.phase = "items"
            self._add_new_item()
            return "Artikelnummer? (Pflicht)"
        return "Bitte gib die angefragte Information an."

    def _handle_items(self, value: str) -> str:
        it = self._current_item()
        if it is None:
            self._add_new_item(); it = self._current_item()
        if self.state.ask_field == "artikelnummer":
            if not value:
                return "Artikelnummer ist erforderlich."
            it.artikelnummer = value
            self.state.ask_field = "menge"
            return "Menge? (Pflicht)"
        if self.state.ask_field == "menge":
            if not value:
                return "Menge ist erforderlich."
            it.menge = value
            # next
            self.state.phase = "issue"
            self.state.ask_field = "kategorie"
            return "Kategorie? (z. B. defekt/beschädigt/falsch geliefert)"
        return "Bitte gib die angefragte Information an."

    def _handle_issue(self, value: str) -> str:
        if self.state.ask_field == "kategorie":
            self.state.kategorie = value or "defekt"
            self.state.ask_field = "beschreibung"
            return "Kurzbeschreibung des Problems?"
        if self.state.ask_field == "beschreibung":
            self.state.beschreibung = value or ""
            self.state.phase = "evidence"
            self.state.ask_field = "evidence"
            return "Bitte Links/Dateinamen zu Fotos/Videos angeben (optional, mehrere durch Komma)."
        return "Bitte gib die angefragte Information an."

    def _handle_evidence(self, value: str) -> str:
        if self.state.ask_field == "evidence":
            self.state.evidence_links = [s.strip() for s in (value or "").split(",") if s.strip()]
            self.state.phase = "preference"
            self.state.ask_field = "preferred"
            return "Bevorzugtes Vorgehen? (repair/replace/refund/return)"
        return "Bitte gib die angefragte Information an."

    def _handle_preference(self, value: str) -> str:
        if self.state.ask_field == "preferred":
            v = (value or "").strip().lower()
            if v not in {"repair", "replace", "refund", "return", "reparatur", "ersatz", "gutschrift", "rueckgabe", "rückgabe"}:
                return "Bitte repair/replace/refund/return angeben (oder deutsches Äquivalent)."
            mapping = {
                "reparatur": "repair", "ersatz": "replace", "gutschrift": "refund", "rueckgabe": "return", "rückgabe": "return"
            }
            self.state.preferred_action = mapping.get(v, v)
            # build summary
            self.state.summary_cache = self._build_summary()
            self.state.phase = "confirm"
            self.state.ask_field = None
            return self.state.summary_cache + "\n\nBitte bestätigen: **ja** / **nein**"
        return "Bitte gib die angefragte Information an."

    def _build_summary(self) -> str:
        s = self.state
        lines: List[str] = ["Zusammenfassung der Reklamation (RMA):"]
        cs = s.customer
        lines += [
            f"E-Mail: {cs.email}",
            f"Kundennummer: {cs.kundennummer or '-'}",
            f"Typ/Name: {cs.typ or '-'} / {cs.name or '-'}",
            f"Auftragsnr.: {s.auftragsnr or '-'} | Rechnungsnr.: {s.rechnungsnr or '-'} | Datum: {s.kaufdatum or '-'}",
        ]
        for i, it in enumerate(s.items, 1):
            if it.is_complete():
                lines.append(f"  {i}) {it.artikelnummer} × {it.menge}")
        lines += [
            f"Kategorie: {s.kategorie}",
            f"Beschreibung: {s.beschreibung}",
            f"Belege: {', '.join(s.evidence_links) if s.evidence_links else '-'}",
            f"Bevorzugt: {s.preferred_action}",
        ]
        return "\n".join(lines)

    def _to_payload(self) -> Dict[str, Any]:
        s = self.state
        return {
            "customer": {
                "email": s.customer.email,
                "kundennummer": s.customer.kundennummer,
                "typ": s.customer.typ,
                "name": s.customer.name,
            },
            "items": [{"artikelnummer": it.artikelnummer, "menge": it.menge} for it in s.items if it.is_complete()],
            "reference": {
                "auftragsnr": s.auftragsnr,
                "rechnungsnr": s.rechnungsnr,
                "kaufdatum": s.kaufdatum,
            },
            "issue": {
                "kategorie": s.kategorie,
                "beschreibung": s.beschreibung,
            },
            "evidence": {
                "links": s.evidence_links,
            },
            "preferred_action": s.preferred_action or "repair",
        }

