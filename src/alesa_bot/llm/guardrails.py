# ===================== FILE: src/alesa_bot/llm/guardrails.py =====================
from __future__ import annotations


def must_have_sources(found_count: int) -> bool:
    """Gibt True zurück, wenn mindestens eine Quelle gefunden wurde, sonst False. Wirft keine Exception mehr."""
    return found_count > 0

