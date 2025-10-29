# src/alesa_bot/services/qa_service.py
from __future__ import annotations
from typing import List, Tuple, Optional
from src.alesa_bot.core.types import Retriever, LLM, Hit
from src.alesa_bot.llm.prompts import build_prompt
from src.alesa_bot.llm.guardrails import must_have_sources
from src.alesa_bot.retrieval.tables import ProductTableStore, ARTICLE_RX
from src.alesa_bot.retrieval.tables import ProductRow

class QAService:
    def __init__(self, retriever: Retriever, llm: LLM, system_prompt: str, query_expand: bool = True,
                 product_store: Optional[ProductTableStore] = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt
        self.query_expand = query_expand
        self.product_store = product_store
        # set on product responses to enable follow-up actions in controller
        self.last_product: Optional[ProductRow] = None

    def _expand(self, question: str) -> str:
        if not self.query_expand:
            return question
        try:
            self.llm.start()
            q = ( "Formuliere 3 kurze Varianten/Synonyme meiner Frage als Stichworte, "
                  "durch Kommas getrennt. Nur die Varianten ausgeben. Frage: " + question )
            variants = (self.llm.generate(q) or "").replace("\n", ", ")
            return f"{question}. Variationen: {variants}"
        except Exception:
            return question

    def ask(self, question: str) -> Tuple[str, List[str]]:
        self.last_product = None
        # 0) Article-number fast path with structured table lookup
        if self.product_store is not None and ARTICLE_RX.search(question or ""):
            pr = self.product_store.find_by_code(question)
            if pr is not None:
                self.last_product = pr
                header = "| d1 | b | b2 | Nuttiefe | d2 | d3 | Zahnform | Aufnahme |\n|---|---|---|---|---|---|---|---|\n"
                row = f"| {pr.d1 or '-'} | {pr.b or '-'} | {pr.b2 or '-'} | {pr.nuttiefe or '-'} | {pr.d2 or '-'} | {pr.d3 or '-'} | {pr.zahnform or '-'} | {pr.aufnahme or '-'} |"
                answer = (f"Maße für Artikel {pr.code_raw}:\n\n" + header + row)
                cite = f"{str(pr.source_path) if pr.source_path else ''}{(' S. '+str(pr.source_page)) if pr.source_page else ''}"
                cites = [c for c in [cite] if c.strip()]
                answer = answer + "\n\nMoechten Sie dieses Produkt bestellen? (Antwort: 'ja' oder 'bestellen')"
                return answer, cites

        q = self._expand(question)
        hits: List[Hit] = self.retriever.search(q, top_k=8)
        must_have_sources(len(hits))
        snippets = [h.snippet for h in hits]
        prompt = build_prompt(self.system_prompt, snippets, question)
        self.llm.start()
        answer = (self.llm.generate(prompt) or '').strip()
        cites = [f"[{i+1}] {h.path}{(' S. '+str(h.page)) if h.page else ''}" for i,h in enumerate(hits)]
        return (answer if answer else "Dafür habe ich in den Dateien keine Quelle gefunden.", cites)
