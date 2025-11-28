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
        self._log = logging.getLogger(__name__)

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
        self.last_product = None
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
                    return self._translate_out(answer, lang_guess.code), cites
                else:
                    msg = "Keine Artikel gefunden für Filter: " + ", ".join(f"{k}={v}" for k, v in filters.items())
                    return self._translate_out(msg, lang_guess.code), []

        q = self._expand(retrieval_question)
        hits: List[Hit] = self.retriever.search(q, top_k=8)
        has_sources = must_have_sources(len(hits))

        if has_sources:
            snippets = [h.snippet for h in hits]
            prompt = build_prompt(self.system_prompt, snippets, retrieval_question, user_lang=lang_guess.code)
            self.llm.start()
            answer = (self.llm.generate(prompt) or "").strip()
            cites = [f"[{i+1}] {h.path}{(' S. ' + str(h.page)) if h.page else ''}" for i, h in enumerate(hits)]
            answer = answer if answer else "Dafuer habe ich in den Dateien keine Quelle gefunden."
            return self._translate_out(answer, lang_guess.code), cites

        # Fallback: generative Antwort ohne Quellen
        if self._log:
            self._log.info("Keine Treffer aus Retrieval – Fallback auf generative Antwort ohne Quellen.")
        prompt = build_prompt(self.system_prompt, [], retrieval_question, user_lang=lang_guess.code)
        self.llm.start()
        answer = (self.llm.generate(prompt) or "").strip()
        answer = answer if answer else "Keine passenden Quellen gefunden, daher generative Antwort ohne Belege."
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
