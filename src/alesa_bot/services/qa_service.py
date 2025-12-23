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
Du bist ALESA, der virtuelle Assistent fuer Produkte, Services und Dokumente von ALESA (Schweizer Saegewerke AG).

Zweck:
- Beantworte Nutzerfragen praezise und freundlich in der Sprache der Frage (Standard: Deutsch).
- Nutze ausschliesslich die gelieferten Snippets/Sources; keine Spekulation oder Halluzination.
- Gib Quellen als [1], [2], ... an. Wenn keine Quelle vorhanden ist, erklaere das offen und bitte ggf. um Praezisierung.
- Fasse dich kurz, nutze Aufzaehlungen bei mehreren Punkten, erklaere Fachbegriffe in Alltagssprache.

Arbeitsweise:
- Kombiniere relevante Snippets, entferne Redundanz, aber behalte wichtige Zahlen, Masse und Produktcodes.
- Wenn etwas unklar bleibt, stelle eine kurze Rueckfrage statt zu raten.
- Keine allgemeinen Aufzaehlungen aus AGB oder Handbuechern; nur Inhalte, die zur Frage passen.
- Wenn die Anfrage ausserhalb des Themenbereichs liegt, erklaere freundlich den Fokus auf ALESA-Themen.

Integriertes Reklamationsprotokoll (aktivieren, wenn Nachricht nach Reklamation/Schaden klingt oder der Reklamationsmodus gesetzt ist):
1) Starte mit kurzer Entschuldigung und Empathie, Beleidigungen ignorieren.
2) Fuehre einen Frage-Antwort-Dialog und sammle schrittweise:
   - Lieferdatum
   - Bestellte Produkte/Artikel mit Mengen
   - Was fehlt/ist defekt (mit Mengenabweichungen)
   - Referenznummern (AB, Bestellung, Lieferschein/LS; alle als gueltig akzeptieren)
   - Name der Kontaktperson
   - Telefonnummer
   - Gewuenschtes Zeitfenster fuer Rueckruf
   - Optional: Hinweis auf Fotos vom Schaden
3) Pro Nutzerantwort genau eine neue Frage stellen; nichts doppelt fragen; kurze Antworten als Fortschritt akzeptieren.
4) Pruefe knapp, ob Reklamationsfrist (z. B. 8 Tage ab Zustellung) abgelaufen sein koennte; auch dann weiterhin hilfsbereit, Daten aufnehmen und Weiterleitung anbieten.
5) Keine Versprechen oder finalen Zusagen (kein "ich erstatte", "ich liefere nach"); du sammelst nur Informationen und leitest intern weiter.
6) Bleibe im Chat, nenne Kontaktwege nur ergaenzend am Ende.
7) Abschluss: Alle gesammelten Daten in Stichpunkten zusammenfassen, fuer den Hinweis danken, bestaetigen, dass der Fall intern geprueft und jemand sich meldet.
"""

COMPLAINT_SYSTEM_PROMPT = """
Du bist im Reklamationsmodus.

- Folge strikt dem integrierten Reklamationsprotokoll aus dem Basis-Prompt.
- Fokus auf Dialog statt langer RAG-Ausgaben oder AGB-Listen; eine Frage pro Antwort, keine Dopplungen.
- Akzeptiere AB-, LS-, Bestell- oder Lieferscheinnummern als Referenzen und gehe direkt zur naechsten fehlenden Information.
- Wenn Frist vermutlich abgelaufen: kurz erklaeren, trotzdem Daten aufnehmen und Weiterleitung anbieten.
- Abschluss: kompakte Stichpunkte mit allen Angaben, Hinweis auf interne Weiterleitung und Rueckmeldung.
"""

# Profile: erweiterbar um weitere Modi
PROMPT_PROFILES: dict[str, list[str]] = {
    "default": [],
    "complaint": [COMPLAINT_SYSTEM_PROMPT],
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
        self.history: List[Tuple[str, str]] = []
        self._empathy_instruction = (
            "Reagiere immer freundlich und empathisch, insbesondere bei Reklamationen "
            "oder verÃ¤rgerten Nachrichten. ErklÃ¤re Regeln (z. B. AGB, Garantie, Fristen) "
            "in verstÃ¤ndlicher Alltagssprache und biete konkrete nÃ¤chste Schritte an. "
            "Ignoriere Beleidigungen und spiegele sie nicht zurÃ¼ck."
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

    def _record_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Merkt sich den bisherigen Chatverlauf (gekuerzt)."""
        try:
            self.history.append(("user", user_msg))
            self.history.append(("assistant", assistant_msg))
            if len(self.history) > 20:
                self.history = self.history[-20:]
        except Exception:
            if self._log:
                self._log.debug("Konnte Chat-Historie nicht speichern", exc_info=True)

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
        history_context = self.history[-12:]

        if self.lang_helper:
            lang_guess, retrieval_question = self.lang_helper.prepare_query(question)
        self.last_lang = lang_guess.code

        def _finish(ans: str, cites: List[str]) -> Tuple[str, List[str]]:
            final = self._translate_out(ans, lang_guess.code)
            self._record_turn(question, final)
            return final, cites

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
                    # dynamischer Headername fÃ¼r Aufnahme: original CSV-Label falls vorhanden
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

                return _finish(answer, cites)

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
                    return _finish(answer, cites)
                else:
                    msg = "Keine Artikel gefunden fÃ¼r Filter: " + ", ".join(f"{k}={v}" for k, v in filters.items())
                    return _finish(msg, [])

        q = self._expand(retrieval_question)
        hits: List[Hit] = self.retriever.search(q, top_k=8)
        # Im Reklamationsmodus keinen generischen Fallback zulassen und Kontext ggf. reduzieren
        if is_complaint:
            hits = []  # Dialogfokus behalten; keine AGB-Lawine
        has_sources = must_have_sources(len(hits))

        if has_sources:
            snippets = [h.snippet for h in hits]
            sys_prompts = build_system_prompts(base_prompt=self.system_prompt, mode=mode, empathy=self._empathy_instruction)
            prompt = build_prompt("\n\n".join(sys_prompts), snippets, retrieval_question, user_lang=lang_guess.code, history=history_context)
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
            return _finish(answer, cites)

        # Fallback: generative Antwort ohne Quellen
        if self._log:
            self._log.info("Keine Treffer aus Retrieval â€“ Fallback auf generative Antwort ohne Quellen.")
        sys_prompts = build_system_prompts(base_prompt=self.system_prompt, mode=mode, empathy=self._empathy_instruction)
        prompt = build_prompt("\n\n".join(sys_prompts), [], retrieval_question, user_lang=lang_guess.code, history=history_context)
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
        return _finish(answer, [])


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
        # Werte bereinigen: abschlieÃŸende Kommas/Punkte/AnfÃ¼hrungszeichen/Einheiten entfernen
        val = raw_val.strip()
        val = val.rstrip(",.;")
        val = val.replace("mm", "").replace(" ", "")
        val = val.replace('"', '').replace("'", "")
        # nur die fÃ¼hrende Zahl/Range extrahieren (z. B. "32angeben?" -> "32")
        m_num = re.match(r"[0-9][0-9.,]*", val)
        if m_num:
            val = m_num.group(0)
        if not val:
            continue
        filters[canon] = val
    return filters


def _clean_val(val: str) -> str:
    """Trimmt Zellwerte und entfernt fÃ¼hrende/abschlieÃŸende Quotes sowie NULL."""
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


# ------- Moduserkennung -------

def detect_mode(user_text: str) -> str:
    """
    Einfache Modus-Erkennung anhand der aktuellen Nachricht.
    Wird vor allem fuer neue Konversationen genutzt; bestehende behalten ihren Modus.
    """
    low = (user_text or "").lower()
    complaint_terms = [
        "kaputt",
        "defekt",
        "unvollstÃ¤ndig",
        "unvollstaendig",
        "reklamation",
        "falsch geliefert",
        "funktioniert nicht",
        "beschÃ¤digt",
        "beschaedigt",
        "ware ist kaputt",
        "ware ist unvollstÃ¤ndig",
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
    # Profile enthalten bereits BASE_SYSTEM_PROMPT; nur zusÃ¤tzliche Elemente anhÃ¤ngen
    for p in profile_prompts:
        if p != BASE_SYSTEM_PROMPT:  # vermeidet doppelte Basis
            stack.append(p)
    if empathy:
        stack.append(empathy)
    return stack


def is_complaint_or_angry(text: str) -> bool:
    """
    Einfache Heuristik zur Erkennung von Reklamationen/VerÃ¤rgerung.
    Beleidigungen werden nur erkannt, nicht gespiegelt.
    """
    low = (text or "").lower()
    complaint_terms = [
        "kaputt",
        "defekt",
        "funktioniert nicht",
        "beschaedigt",
        "beschÃ¤digt",
        "reklamation",
        "garantie",
        "gewaehrleistung",
        "gewÃ¤hrleistung",
        "retoure",
        "umtausch",
    ]
    negative_terms = [
        "scheisse",
        "scheiÃŸe",
        "schweine",
        "mies",
        "verarsche",
        "unzufrieden",
        "saftladen",
        "beschwerde",
        "aergerlich",
        "Ã¤rgerlich",
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
    Baut eine empathische, handlungsorientierte Antwort fÃ¼r Reklamationen.
    Beleidigungen werden bewusst nicht zurÃ¼ckgespiegelt.
    """
    lines: List[str] = []
    lines.append("Es tut mir leid, dass deine Lieferung beschÃ¤digt oder defekt ist. Das ist Ã¤rgerlich.")
    lines.append("Ich verstehe den Frust und helfe dir sofort, das zu klÃ¤ren.")

    # Aus der RAG-Antwort oder Snippets kurze Hinweise ziehen
    bullet_candidates: List[str] = []
    for part in (rag_answer or "").splitlines():
        p = part.strip().lstrip("-â€¢").strip()
        if p:
            bullet_candidates.append(p)
    if not bullet_candidates:
        bullet_candidates = legal_snippets[:]
    if not bullet_candidates:
        bullet_candidates = [
            "Reklamationen sind innerhalb der Ã¼blichen Frist (z. B. 8 Tage nach Erhalt) mÃ¶glich.",
            "Wir prÃ¼fen, ob Reparatur oder Ersatz sinnvoll ist.",
        ]

    lines.append("Kurz das Wichtigste:")
    for b in bullet_candidates[:4]:
        lines.append(f"- {b}")

    lines.append("Bitte schick mir deine Bestell- oder Lieferscheinnummer und Fotos vom Schaden, dann starte ich den Reklamationsprozess.")
    return "\n".join(lines)
