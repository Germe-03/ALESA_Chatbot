# src/alesa_bot/services/qa_service.py
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from src.alesa_bot.core.types import Hit, LLM, Retriever
from src.alesa_bot.llm.guardrails import must_have_sources
from src.alesa_bot.llm.prompts import build_prompt
from src.alesa_bot.retrieval.tables import ARTICLE_RX, ProductRow, ProductTableStore
from src.alesa_bot.services.language import LangGuess, LanguageHelper

# ---- System-Prompts (Basis + Reklamationsmodus) ----
BASE_SYSTEM_PROMPT = """
Hallo! Ich bin ALESA, dein virtueller KI-Assistent.

Stell mir eine Frage zu Produkten, Services oder Dokumenten – ich antworte mit Quellen.
"""

COMPLAINT_SYSTEM_PROMPT = """
Du bist im Modus 'Reklamationsassistent' für ALESA.

Allgemein:
- Bleib immer höflich, ruhig und professionell, auch wenn der Nutzer wütend oder beleidigend ist.
- Ignoriere Beleidigungen und spiegle sie NICHT zurück.
- Erkläre Regeln (z.B. AGB) in einfacher Alltagssprache und nur so ausführlich wie nötig.
- Gib NIEMALS eine finale Lösung oder Zusage (kein „ich liefere nach“, „ich erstatte“); du sammelst nur Infos und leitest weiter.

Ziel in diesem Modus:
- Du sollst im Chat alle wichtigen Informationen für eine Reklamation einsammeln.
- Du führst dazu einen kurzen Frage-Antwort-Dialog.
- Du bleibst IM CHAT, anstatt den Nutzer direkt wegzuschicken.

Sammle schrittweise folgende Informationen:
1. Datum, an dem die Ware beim Kunden angekommen ist.
2. Was wurde bestellt? (Produkt / Artikel / Menge)
3. Was genau ist defekt oder fehlt? (inkl. Mengen, z.B. "10 bestellt, 9 angekommen").
4. AB- oder Lieferscheinnummer (falls vorhanden).
5. Name der Kontaktperson.
6. Telefonnummer.
7. Eine Zeit oder Zeitspanne, wann wir zurückrufen können.

Wichtiger Ablauf:
- Stelle pro Antwort des Nutzers nur EINE neue Frage.
- Stelle KEINE Fragen doppelt, wenn die Information bereits genannt wurde.
- Wenn der Nutzer kurz antwortet („heute“, „02.12.2025 LS452415“, „10 bestellt 9 angekommen“), interpretiere das als Antwort auf deine letzte Frage und gehe zur nächsten Frage über.
- Zitiere AGB nur kurz und nur, wenn es dem Nutzer hilft (z.B. Reklamationsfrist), aber mache keine lange Liste daraus.
- Prüfe, ob eine Reklamation formal noch zulässig wäre (z.B. Frist 8 Tage ab Zustellung). Wenn die Frist unklar ist: nachfragen. Wenn sie abgelaufen ist: freundlich erklären und nur Kontaktwege nennen.

Am Ende, wenn du alle wichtigen Informationen hast (oder der Nutzer keine weiteren Angaben machen möchte):
- Fasse alle Daten der Reklamation übersichtlich in Stichpunkten zusammen.
- Sage, dass du die Reklamation intern weiterleitest und dass sich jemand bei ihm meldet.
- Erst am Ende kannst du optional Kontaktmöglichkeiten nennen (Telefon/E-Mail), aber nicht als Hauptlösung, sondern ergänzend.

Wenn du nach einer Bestell- oder Lieferscheinnummer fragst und der Nutzer stattdessen eine AB-Nummer, LS-Nummer oder ähnliche Referenz angibt (z.B. "AB Nummer VKA421521", "AB VKA...", "LS 123456"), dann:
- akzeptiere diese Nummer als gültige Referenz,
- bedanke dich kurz dafür,
- und stelle direkt die nächste notwendige Frage im Reklamationsdialog (z.B. "Was wurde bestellt und was genau fehlt?" oder "Wie viele Stück wurden bestellt und wie viele sind angekommen?").

Antworten wie "Da du keine spezifische Frage gestellt hast..." oder lange allgemeine Aufzählungen aus den AGB sind im Reklamationsmodus NICHT erlaubt.
Im Reklamationsmodus gilt:
- Bleib im Frage-Antwort-Dialog, bis du alle nötigen Infos hast (Ankunftsdatum, was bestellt / was fehlt, Mengen, Referenznummern, Name, Telefonnummer, Rückrufzeit).
- Verwende Informationen aus den AGB nur kurz und gezielt (z.B. Frist: "innerhalb von 8 Tagen"), aber mache keine lange Liste mit vielen Punkten.
"""

# Profile: erweiterbar um weitere Modi
PROMPT_PROFILES: dict[str, list[str]] = {
    "default": [BASE_SYSTEM_PROMPT],
    "complaint": [BASE_SYSTEM_PROMPT, COMPLAINT_SYSTEM_PROMPT],
}


class QAService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        system_prompt: str,
        query_expand: bool = True,
        product_store: Optional[ProductTableStore] = None,
        lang_helper: Optional[LanguageHelper] = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt
        self.query_expand = query_expand
        self.product_store = product_store
        self.lang_helper = lang_helper
        # set on product responses to enable follow-up actions in controller
        self.last_product: Optional[ProductRow] = None
        # remembers Sprache der letzten Anfrage (z. B. fuer UI oder Tests)
        self.last_lang: str = "de"
        self.last_sources: List[str] = []
        self.last_keywords: List[str] = []
        self._log = logging.getLogger(__name__)
        self._empathy_instruction = (
            "Reagiere immer freundlich und empathisch, insbesondere bei Reklamationen "
            "oder verärgerten Nachrichten. Erkläre Regeln (z. B. AGB, Garantie, Fristen) "
            "in verständlicher Alltagssprache und biete konkrete nächste Schritte an. "
            "Ignoriere Beleidigungen und spiegele sie nicht zurück."
        )

        # ASCII-freundliche Variante, falls Encoding der obigen Zeilen fehlschlaegt
        self._empathy_instruction = (
            "Reagiere immer freundlich und empathisch, insbesondere bei Reklamationen "
            "oder veraergerten Nachrichten. Erklaere Regeln (z. B. AGB, Garantie, Fristen) "
            "in verstaendlicher Alltagssprache und biete konkrete naechste Schritte an. "
            "Ignoriere Beleidigungen und spiegele sie nicht zurueck."
        )
        # Modus pro Konversation (default/complaint); wird einmalig gesetzt und bleibt bis Ende der Session
        self.conversation_mode: str = ""

    # -------- interne Helfer --------

    def _expand(self, question: str) -> str:
        if not self.query_expand:
            return question
        try:
            self.llm.start()
            q = (
                "Formuliere 3 kurze Varianten/Synonyme meiner Frage als Stichworte, "
                "durch Kommas getrennt. Nur die Varianten ausgeben. Frage: " + question
            )
            variants = (self.llm.generate(q) or "").replace("\n", ", ")
            return f"{question}. Variationen: {variants}"
        except Exception:
            return question

    def _translate_out(self, answer: str, target_lang: str) -> str:
        if not self.lang_helper:
            return answer
        return self.lang_helper.render_answer(answer, target_lang)

    # -------- oeffentliche API --------

    def ask(self, question: str) -> Tuple[str, List[str]]:
        self.last_sources = []
        self.last_keywords = []
        self.last_product = None
        # Modus-Erkennung pro Konversation (bleibt erhalten, auch wenn Folge-Nachrichten neutral sind)
        mode = get_conversation_mode(self, question)
        is_complaint = (mode == "complaint") or is_complaint_or_angry(question)
        lang_guess: LangGuess = LangGuess(code="de", confidence=1.0)
        retrieval_question = question

        if self.lang_helper:
            lang_guess, retrieval_question = self.lang_helper.prepare_query(question)
        self.last_lang = lang_guess.code

        # 0) Article-number fast path with structured table lookup
        if self.product_store is not None and ARTICLE_RX.search(question or ""):
            prs = self.product_store.find_all_in_text(question)
            if not prs:
                pr_single = self.product_store.find_by_code(question)
                prs = [pr_single] if pr_single else []
            if prs:
                self.last_product = prs[0]

                # group by familie/gruppe/fallback
                groups: dict[str, List[ProductRow]] = {}
                for pr in prs:
                    key = (pr.gruppe or (pr.source_path.stem if pr.source_path else "") or pr.code_norm[:4] or "Artikel").strip()
                    groups.setdefault(key, []).append(pr)

                cols_def = [
                    ("Artikel", lambda p: p.code_raw),
                    ("Gruppe", lambda p: p.gruppe or "-"),
                    ("d1", lambda p: _clean_val(p.d1)),
                    ("b", lambda p: _clean_val(p.b)),
                    ("b2", lambda p: _clean_val(p.b2)),
                    ("Nuttiefe", lambda p: _clean_val(p.nuttiefe)),
                    ("d2", lambda p: _clean_val(p.d2)),
                    ("d3", lambda p: _clean_val(p.d3)),
                    ("d4", lambda p: _clean_val(p.d4)),
                    ("Saege-O", lambda p: _clean_val(p.saegen_o)),
                    ("L", lambda p: _clean_val(p.l)),
                    ("l1", lambda p: _clean_val(p.l1)),
                    ("l2", lambda p: _clean_val(p.l2)),
                    ("G", lambda p: _clean_val(p.g)),
                    ("Aufnahme", lambda p: _clean_val(p.aufnahme)),
                ]

                lines: List[str] = []
                all_codes = ", ".join(pr.code_raw for pr in prs)
                lines.append(f"Artikel: {all_codes}")
                for gname, items in groups.items():
                    lines.append(f"\nGruppe: {gname}")
                    # determine columns that have data in this group
                    active_cols = []
                    for label, getter in cols_def:
                        vals = [getter(p) for p in items]
                        if any(v for v in vals):
                            active_cols.append((label, getter))
                    if not active_cols:
                        continue
                    # dynamischer Headername für Aufnahme: original CSV-Label falls vorhanden
                    def _label_for(col_label: str) -> str:
                        if col_label != "Aufnahme":
                            return col_label
                        raw_label = next((getattr(p, "aufnahme_label", "") for p in items if getattr(p, "aufnahme_label", "")), "")
                        return raw_label or "Aufnahme"

                    header = "| " + " | ".join(_label_for(l) for l, _ in active_cols) + " |"
                    sep = "| " + " | ".join("---" for _ in active_cols) + " |"
                    rows = []
                    for p in items:
                        rows.append("| " + " | ".join(getter(p) or "-" for _, getter in active_cols) + " |")
                    lines.extend([header, sep, *rows])

                answer = "\n".join(lines)
                cites_set = set()
                for pr in prs:
                    cite = f"{str(pr.source_path) if pr.source_path else ''}{(' S. ' + str(pr.source_page)) if pr.source_page else ''}"
                    if cite.strip():
                        cites_set.add(cite.strip())
                cites = list(cites_set)
                self.last_sources = cites
                self.last_keywords = [p.code_norm for p in prs if getattr(p, "code_norm", None)]

                if _has_order_intent(question):
                    answer = answer + "\n\nMoechten Sie dieses Produkt bestellen? (Antwort: 'ja' oder 'bestellen')"

                return self._translate_out(answer, lang_guess.code), cites

        # 0b) Filter-basierte Produktsuche (z. B. "suche artikel mit d1: 40, b: 1.0, nuttiefe: 14.5")
        if self.product_store is not None:
            filters = _parse_product_filters(question)
            if filters:
                prs = self.product_store.filter_rows(filters)
                if prs:
                    lines: List[str] = []
                    lines.append(f"Gefundene Artikel: {', '.join(p.code_raw for p in prs)}")
                    cols_def = [
                        ("Artikel", lambda p: p.code_raw),
                        ("Gruppe", lambda p: p.gruppe or "-"),
                        ("d1", lambda p: _clean_val(p.d1)),
                        ("b", lambda p: _clean_val(p.b)),
                        ("b2", lambda p: _clean_val(p.b2)),
                        ("Nuttiefe", lambda p: _clean_val(p.nuttiefe)),
                        ("d2", lambda p: _clean_val(p.d2)),
                        ("d3", lambda p: _clean_val(p.d3)),
                        ("Aufnahme", lambda p: _clean_val(p.aufnahme)),
                    ]
                    active_cols = []
                    for label, getter in cols_def:
                        vals = [getter(p) for p in prs]
                        if any(v for v in vals):
                            active_cols.append((label, getter))
                    if active_cols:
                        header = "| " + " | ".join(l for l, _ in active_cols) + " |"
                        sep = "| " + " | ".join("---" for _ in active_cols) + " |"
                        lines.append("\n" + header)
                        lines.append(sep)
                        for p in prs:
                            lines.append("| " + " | ".join(getter(p) or "-" for _, getter in active_cols) + " |")
                    answer = "\n".join(lines)
                    cites: List[str] = []
                    self.last_sources = cites
                    self.last_keywords = [p.code_norm for p in prs if getattr(p, "code_norm", None)]
                    return self._translate_out(answer, lang_guess.code), cites
                else:
                    msg = "Keine Artikel gefunden für Filter: " + ", ".join(f"{k}={v}" for k, v in filters.items())
                    return self._translate_out(msg, lang_guess.code), []

        q = self._expand(retrieval_question)
        hits: List[Hit] = self.retriever.search(q, top_k=8)
        # Im Reklamationsmodus keinen generischen Fallback zulassen und Kontext ggf. reduzieren
        if is_complaint:
            hits = []  # Dialogfokus behalten; keine AGB-Lawine
        has_sources = must_have_sources(len(hits))

        if has_sources:
            snippets = [h.snippet for h in hits]
            sys_prompts = build_system_prompts(base_prompt=self.system_prompt, mode=mode, empathy=self._empathy_instruction)
            prompt = build_prompt("\n\n".join(sys_prompts), snippets, retrieval_question, user_lang=lang_guess.code)
            self.llm.start()
            answer = (self.llm.generate(prompt) or "").strip()
            cites = [f"[{i+1}] {h.path}{(' S. ' + str(h.page)) if h.page else ''}" for i, h in enumerate(hits)]
            answer = answer if answer else "Dafuer habe ich in den Dateien keine Quelle gefunden."
            if is_complaint:
                answer = build_complaint_response(
                    user_message=question,
                    rag_answer=answer,
                    legal_snippets=[s for s in snippets if s],
                    sources=cites,
                )
            self.last_sources = cites
            self.last_keywords = [h.path for h in hits if getattr(h, "path", None)]
            return self._translate_out(answer, lang_guess.code), cites

        # Fallback: generative Antwort ohne Quellen
        if self._log:
            self._log.info("Keine Treffer aus Retrieval – Fallback auf generative Antwort ohne Quellen.")
        sys_prompts = build_system_prompts(base_prompt=self.system_prompt, mode=mode, empathy=self._empathy_instruction)
        prompt = build_prompt("\n\n".join(sys_prompts), [], retrieval_question, user_lang=lang_guess.code)
        self.llm.start()
        answer = (self.llm.generate(prompt) or "").strip()
        answer = answer if answer else "Keine passenden Quellen gefunden, daher generative Antwort ohne Belege."
        if is_complaint:
            answer = build_complaint_response(
                user_message=question,
                rag_answer=answer,
                legal_snippets=[],
                sources=[],
            )
        self.last_sources = []
        self.last_keywords = []
        return self._translate_out(answer, lang_guess.code), []


def _has_order_intent(text: str) -> bool:
    low = (text or '').lower()
    intents = ["bestellen", "kaufen", "order", "ich moechte", "bitte bestellen"]
    return any(t in low for t in intents)


def _parse_product_filters(text: str) -> dict:
    """
    Extrahiert einfache Filter aus Freitext wie 'd1: 40, b: 1.0, nuttiefe: 14.5'.
    Robust gegen Kommas/Punkte/Einheiten (mm) und Synonyme (b1->b2, t->nuttiefe).
    """
    if not text:
        return {}
    t = text.lower()
    # erlaubte Keys + Synonyme
    key_map = {
        "d1": "d1",
        "d1mm": "d1",
        "b": "b",
        "b1": "b2",
        "b2": "b2",
        "nuttiefe": "nuttiefe",
        "nut": "nuttiefe",
        "t": "nuttiefe",
        "d2": "d2",
        "d3": "d3",
        "d4": "d4",
        "aufnahme": "aufnahme",
        "gruppe": "gruppe",
    }
    # finde Paare key: value mit : oder =
    pattern = re.compile(r"(d1mm|d1|b1|b2|b|nuttiefe|nut|t|d2|d3|d4|aufnahme|gruppe)\s*[:=]\s*([^,;\n]+)")
    filters: dict[str, str] = {}
    for m in pattern.finditer(t):
        raw_key = m.group(1)
        raw_val = m.group(2)
        canon = key_map.get(raw_key)
        if not canon:
            continue
        # Werte bereinigen: abschließende Kommas/Punkte/Anführungszeichen/Einheiten entfernen
        val = raw_val.strip()
        val = val.rstrip(",.;")
        val = val.replace("mm", "").replace(" ", "")
        val = val.replace('"', '').replace("'", "")
        # nur die führende Zahl/Range extrahieren (z. B. "32angeben?" -> "32")
        m_num = re.match(r"[0-9][0-9.,]*", val)
        if m_num:
            val = m_num.group(0)
        if not val:
            continue
        filters[canon] = val
    return filters


def _clean_val(val: str) -> str:
    """Trimmt Zellwerte und entfernt führende/abschließende Quotes sowie NULL."""
    if val is None:
        return ""
    v = val.strip()
    if not v:
        return ""
    if v.startswith('"') and v.endswith('"') and len(v) >= 2:
        v = v[1:-1].strip()
    if v.upper() == "NULL":
        return "NULL"
    return v


# ------- Prompt-Profile & Moduserkennung -------
# BASE_SYSTEM_PROMPT wird dynamisch aus self.system_prompt gespeist.
COMPLAINT_SYSTEM_PROMPT = """
Du bist im Modus 'Reklamationsassistent'.

Wenn sich der Nutzer über defekte, unvollständige oder falsch gelieferte Ware beschwert
(z.B. Wörter wie 'kaputt', 'defekt', 'unvollständig', 'falsch geliefert', 'Reklamation'):

1. Reagiere zuerst freundlich und entschuldigend.
2. Führe ein kurzes Frage-Antwort-Gespräch, um diese Infos zu sammeln:
   - Datum, wann die Ware angekommen ist
   - Was wurde bestellt und was fehlt / ist defekt (inkl. Mengen, z.B. '10 bestellt, 9 angekommen')
   - AB-Nummer oder Lieferscheinnummer (falls vorhanden)
   - Name, Telefonnummer und eine Zeitspanne für Rückruf
3. Stelle pro Antwort des Nutzers nur EINE neue Frage und stelle nichts doppelt.
4. Am Ende fasse alle Infos in Stichpunkten zusammen und sage, dass du die Reklamation weiterleitest.
5. Bleib immer höflich, ignoriere Beleidigungen, zitiere AGB nur knapp in Alltagssprache.
"""

# Profile erlauben spaetere Erweiterungen (z. B. bestellung/beratung) durch einfaches Hinzufügen.
PROMPT_PROFILES: dict[str, list[str]] = {
    "default": [],  # Basis-Prompt wird dynamisch vorne angefügt
    "complaint": [COMPLAINT_SYSTEM_PROMPT],
}


def detect_mode(user_text: str) -> str:
    """
    Einfache Modus-Erkennung anhand der aktuellen Nachricht.
    Wird vor allem fuer neue Konversationen genutzt; bestehende behalten ihren Modus.
    """
    low = (user_text or "").lower()
    complaint_terms = [
        "kaputt",
        "defekt",
        "unvollständig",
        "unvollstaendig",
        "reklamation",
        "falsch geliefert",
        "funktioniert nicht",
        "beschädigt",
        "beschaedigt",
        "ware ist kaputt",
        "ware ist unvollständig",
    ]
    if any(t in low for t in complaint_terms):
        return "complaint"
    return "default"


def build_system_prompts(base_prompt: str, mode: str, empathy: str | None = None) -> list[str]:
    """
    Stellt die System-Prompts zusammen: Basis + Modus-spezifische + optionale Empathie-Hinweise.
    Basis wird aus dem uebergebenen base_prompt (Konfig) genommen, das Profil steuert Zusatzprompts.
    Der Modus bleibt pro Konversation stabil.
    """
    stack: list[str] = []
    base = base_prompt or BASE_SYSTEM_PROMPT
    stack.append(base)
    profile_prompts = PROMPT_PROFILES.get(mode, PROMPT_PROFILES["default"])
    # Profile enthalten bereits BASE_SYSTEM_PROMPT; nur zusätzliche Elemente anhängen
    for p in profile_prompts:
        if p != BASE_SYSTEM_PROMPT:  # vermeidet doppelte Basis
            stack.append(p)
    if empathy:
        stack.append(empathy)
    return stack


def is_complaint_or_angry(text: str) -> bool:
    """
    Einfache Heuristik zur Erkennung von Reklamationen/Verärgerung.
    Beleidigungen werden nur erkannt, nicht gespiegelt.
    """
    low = (text or "").lower()
    complaint_terms = [
        "kaputt",
        "defekt",
        "funktioniert nicht",
        "beschaedigt",
        "beschädigt",
        "reklamation",
        "garantie",
        "gewaehrleistung",
        "gewährleistung",
        "retoure",
        "umtausch",
    ]
    negative_terms = [
        "scheisse",
        "scheiße",
        "schweine",
        "mies",
        "verarsche",
        "unzufrieden",
        "saftladen",
        "beschwerde",
        "aergerlich",
        "ärgerlich",
        "genervt",
    ]
    return any(t in low for t in complaint_terms + negative_terms)


def get_conversation_mode(conversation: "QAService", last_user_text: str) -> str:
    """
    Liefert den Modus fuer diese Konversation. Wenn bereits gesetzt, bleibt er bestehen,
    damit Reklamationsgespraeche nicht durch neutrale Folgemessages auf default zurueckfallen.
    """
    if getattr(conversation, "conversation_mode", ""):
        return conversation.conversation_mode
    mode = detect_mode(last_user_text)
    conversation.conversation_mode = mode
    return mode


def build_complaint_response(user_message: str, rag_answer: str, legal_snippets: List[str], sources: List[str]) -> str:
    """
    Baut eine empathische, handlungsorientierte Antwort für Reklamationen.
    Beleidigungen werden bewusst nicht zurückgespiegelt.
    """
    lines: List[str] = []
    lines.append("Es tut mir leid, dass deine Lieferung beschädigt oder defekt ist. Das ist ärgerlich.")
    lines.append("Ich verstehe den Frust und helfe dir sofort, das zu klären.")

    # Aus der RAG-Antwort oder Snippets kurze Hinweise ziehen
    bullet_candidates: List[str] = []
    for part in (rag_answer or "").splitlines():
        p = part.strip().lstrip("-•").strip()
        if p:
            bullet_candidates.append(p)
    if not bullet_candidates:
        bullet_candidates = legal_snippets[:]
    if not bullet_candidates:
        bullet_candidates = [
            "Reklamationen sind innerhalb der üblichen Frist (z. B. 8 Tage nach Erhalt) möglich.",
            "Wir prüfen, ob Reparatur oder Ersatz sinnvoll ist.",
        ]

    lines.append("Kurz das Wichtigste:")
    for b in bullet_candidates[:4]:
        lines.append(f"- {b}")

    lines.append("Bitte schick mir deine Bestell- oder Lieferscheinnummer und Fotos vom Schaden, dann starte ich den Reklamationsprozess.")
    return "\n".join(lines)
