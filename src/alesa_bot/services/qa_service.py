# src/alesa_bot/services/qa_service.py
from __future__ import annotations

import logging
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
            pr = self.product_store.find_by_code(question)
            if pr is not None:
                self.last_product = pr
                def _clean(val: str) -> str:
                    if val is None:
                        return ""
                    v = val.strip()
                    if not v:
                        return ""
                    if v.upper() == "NULL":
                        return "NULL"
                    return v

                cols = [
                    ("d1", _clean(pr.d1)),
                    ("b", _clean(pr.b)),
                    ("b2", _clean(pr.b2)),
                    ("Nuttiefe", _clean(pr.nuttiefe)),
                    ("d2", _clean(pr.d2)),
                    ("d3", _clean(pr.d3)),
                    ("d4", _clean(pr.d4)),
                    ("Saege-O", _clean(pr.saegen_o)),
                    ("G", _clean(pr.g)),
                    ("l1", _clean(pr.l1)),
                    ("l2", _clean(pr.l2)),
                    ("L", _clean(pr.l)),
                    ("Zahnform", _clean(pr.zahnform)),
                    ("Aufnahme", _clean(pr.aufnahme)),
                ]
                cols = [(h, v) for h, v in cols if v or v == "NULL"]
                if not cols:
                    answer = f"Keine Masse fuer Artikel {pr.code_raw} gefunden."
                    return self._translate_out(answer, lang_guess.code), []

                header = "| " + " | ".join(h for h, _ in cols) + " |\n"
                header += "| " + " | ".join("---" for _ in cols) + " |\n"
                row = "| " + " | ".join(v for _, v in cols) + " |"
                answer = f"Mass fuer Artikel {pr.code_raw}:\n\n{header}{row}"
                cite = f"{str(pr.source_path) if pr.source_path else ''}{(' S. ' + str(pr.source_page)) if pr.source_page else ''}"
                cites = [c for c in [cite] if c.strip()]
                answer = answer + "\n\nMoechten Sie dieses Produkt bestellen? (Antwort: 'ja' oder 'bestellen')"
                return self._translate_out(answer, lang_guess.code), cites

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
