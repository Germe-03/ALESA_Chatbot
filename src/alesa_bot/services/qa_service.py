# ===================== FILE: src/alesa_bot/services/qa_service.py =====================
from __future__ import annotations
from typing import List, Tuple
from src.alesa_bot.core.types import Retriever, LLM, Hit
from src.alesa_bot.llm.prompts import build_prompt
from src.alesa_bot.llm.guardrails import must_have_sources


class QAService:
    def __init__(self, retriever: Retriever, llm: LLM, system_prompt: str) -> None:
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt

    def ask(self, question: str) -> Tuple[str, List[str]]:
        hits: List[Hit] = self.retriever.search(question, top_k=6)
        must_have_sources(len(hits))
        snippets = [h.snippet for h in hits]
        prompt = build_prompt(self.system_prompt, snippets, question)
        self.llm.start()
        answer = (self.llm.generate(prompt) or '').strip()
        cites = [
            f"[{i+1}] {h.path}{(' S. '+str(h.page)) if h.page else ''}"
            for i, h in enumerate(hits)
        ]
        return (answer if answer else "(kein Ergebnis)", cites)
