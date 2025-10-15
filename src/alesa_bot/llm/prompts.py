# src/alesa_bot/llm/prompts.py
from __future__ import annotations
from typing import List

def build_prompt(system_prompt: str, snippets: List[str], question: str) -> str:
    ctx = "\n\n---\n\n".join(snippets)
    guidance = (
        f"{system_prompt}\n"
        "ANTWORT-RICHTLINIEN:\n"
        "1) Antworte NUR auf Basis der Auszüge.\n"
        "2) Decke ALLE gefragten Punkte ab. Wenn ein Punkt nicht belegt ist,\n"
        "   markiere ihn exakt mit: [Keine Quelle in den Dateien gefunden].\n"
        "3) Gib die Antwort als nummerierte Liste aus (1., 2., 3., ...)."
    )
    return (
        f"{guidance}\n\n"
        f"AUSZÜGE BEGINN\n{ctx}\nAUSZÜGE ENDE\n\n"
        f"FRAGE: {question}\n"
        f"ANTWORT:"
    )
