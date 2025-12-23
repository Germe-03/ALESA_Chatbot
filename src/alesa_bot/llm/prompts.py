# src/alesa_bot/llm/prompts.py
from __future__ import annotations
from typing import List, Optional, Tuple


def build_prompt(
    system_prompt: str,
    snippets: List[str],
    question: str,
    user_lang: str = "de",
    history: Optional[List[Tuple[str, str]]] = None,
) -> str:
    ctx = "\n\n---\n\n".join(snippets)
    lang_hint = "Alle Quelldokumente sind deutsch. Formuliere die Fachantwort auf Deutsch, sachlich und praezise."
    if user_lang and user_lang != "de":
        lang_hint += (
            f" Die Nutzerfrage war in Sprache '{user_lang}'. "
            "Auch dann im Deutschen bleiben, damit die Fakten exakt den Quellen folgen."
        )
    guidance = (
        f"{system_prompt}\n"
        f"{lang_hint}\n"
        "ANTWORT-RICHTLINIEN:\n"
        "1) Antworte NUR auf Basis der Auszuege.\n"
        "2) Decke ALLE gefragten Punkte ab; mache klar, wenn etwas in den Auszuegen nicht steht.\n"
        "3) Gib die Antwort als nummerierte Liste aus (1., 2., 3., ...)."
    )
    return (
        f"{guidance}\n\n"
        f"{_format_history(history)}"
        f"AUSZUEGE BEGINN\n{ctx}\nAUSZUEGE ENDE\n\n"
        f"FRAGE: {question}\n"
        f"ANTWORT:"
    )


def _format_history(history: Optional[List[Tuple[str, str]]]) -> str:
    if not history:
        return ""
    lines = ["BISHERIGER CHAT (kurz, chronologisch):"]
    for role, content in history:
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines) + "\n\n"
