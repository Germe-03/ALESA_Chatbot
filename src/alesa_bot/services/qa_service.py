# src/alesa_bot/services/qa_service.py
from __future__ import annotations
from typing import List, Tuple
from src.alesa_bot.core.types import Retriever, LLM, Hit
from src.alesa_bot.llm.prompts import build_prompt
from src.alesa_bot.llm.guardrails import must_have_sources

class QAService:
    def __init__(self, retriever: Retriever, llm: LLM, system_prompt: str, query_expand: bool = True) -> None:
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt
        self.query_expand = query_expand

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
        q = self._expand(question)
        hits: List[Hit] = self.retriever.search(q, top_k=8)
        must_have_sources(len(hits))
        snippets = [h.snippet for h in hits]
        prompt = build_prompt(self.system_prompt, snippets, question)
        self.llm.start()
        answer = (self.llm.generate(prompt) or '').strip()
        cites = [f"[{i+1}] {h.path}{(' S. '+str(h.page)) if h.page else ''}" for i,h in enumerate(hits)]
        return (answer if answer else "Dafür habe ich in den Dateien keine Quelle gefunden.", cites)
