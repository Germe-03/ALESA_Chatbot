# ===================== FILE: src/alesa_bot/llm/prompts.py =====================
from __future__ import annotations
from typing import List


def build_prompt(system_prompt: str, snippets: List[str], question: str) -> str:
    ctx = "\n\n".join(snippets)
    guidance = (
        f"{system_prompt}\n"
        "Beantworte NUR basierend auf den folgenden Auszügen. "
        "Wenn die Auszüge nicht ausreichen, antworte: 'Dafür habe ich in den Dateien keine Quelle gefunden.' "
        "Hänge am Ende eine Liste 'Quellen: [1] …' an."
    )
    return f"{guidance}\n\nAUSZÜGE BEGINN\n{ctx}\nAUSZÜGE ENDE\n\nFRAGE: {question}"
