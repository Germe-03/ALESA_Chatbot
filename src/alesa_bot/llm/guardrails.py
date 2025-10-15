# ===================== FILE: src/alesa_bot/llm/guardrails.py =====================
from __future__ import annotations


def must_have_sources(found_count: int) -> None:
    if found_count <= 0:
        raise ValueError("Keine passenden Quellen gefunden.")
