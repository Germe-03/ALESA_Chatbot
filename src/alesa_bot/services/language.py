from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from langdetect import DetectorFactory, detect_langs

from src.alesa_bot.core.types import LLM

# Deterministisches Ergebnis der Spracherkennung
DetectorFactory.seed = 42

# Unterstuetzte Nutzer-Sprachen (ISO-639-1 Code -> lesbarer Name)
SUPPORTED_LANGS = {
    "de": "Deutsch",
    "en": "English",
    "fr": "Francais",
    "it": "Italiano",
}


@dataclass
class LangGuess:
    code: str
    confidence: float


def detect_language(text: str, threshold: float = 0.6) -> LangGuess:
    """
    Erkennt de/en/fr/it. Bei unsicherem Ergebnis oder anderer Sprache faellt auf Deutsch zurueck.
    """
    try:
        candidates = detect_langs(text or "")
        if not candidates:
            return LangGuess("de", 0.0)
        best = candidates[0]
        code = best.lang.lower()
        conf = float(getattr(best, "prob", 0.0))
    except Exception:
        return LangGuess("de", 0.0)

    if code not in SUPPORTED_LANGS or conf < threshold:
        return LangGuess("de", conf)
    return LangGuess(code, conf)


class LanguageHelper:
    """
    Bietet Spracherkennung und LLM-gestuetzte Uebersetzungen fuer de/en/fr/it.
    Generiert immer eine deutschsprachige Basisantwort (Quellensprache), uebersetzt aber
    fuer den Nutzer, falls die erkannte Sprache nicht Deutsch ist.
    """

    def __init__(self, llm: LLM, min_confidence: float = 0.6) -> None:
        self.llm = llm
        self.min_confidence = min_confidence

    # ---- Erkennung ----

    def guess(self, text: str) -> LangGuess:
        return detect_language(text, threshold=self.min_confidence)

    # ---- Uebersetzungen ----

    def to_german(self, text: str, source_lang: str) -> str:
        if source_lang == "de":
            return text
        if not text:
            return ""
        prompt = (
            "Uebersetze den folgenden Inhalt ins Deutsche. "
            "Erhalte Fachbegriffe und Eigennamen unveraendert. "
            "Nur die Uebersetzung ausgeben, ohne Zusaetze.\n\n"
            f"Quellsprache: {SUPPORTED_LANGS.get(source_lang, source_lang)}\n"
            f"Text: {text}"
        )
        try:
            self.llm.start()
            translated = (self.llm.generate(prompt) or "").strip()
            return translated or text
        except Exception:
            return text

    def from_german(self, text: str, target_lang: str) -> str:
        if target_lang == "de":
            return text
        if not text:
            return ""
        prompt = (
            "Uebersetze den folgenden deutschsprachigen Antworttext "
            f"in {SUPPORTED_LANGS.get(target_lang, target_lang)}. "
            "Erhalte Listenformatierung, Markdown-Tabellen und Referenzmarker wie [1] unveraendert. "
            "Keine neuen Fakten, keine Quellen erfinden. "
            "Nur die Uebersetzung ausgeben.\n\n"
            f"Text: {text}"
        )
        try:
            self.llm.start()
            translated = (self.llm.generate(prompt) or "").strip()
            return translated or text
        except Exception:
            return text

    # ---- Komfort-Helper ----

    def prepare_query(self, question: str) -> Tuple[LangGuess, str]:
        """
        Liefert (Spracherkennung, deutschsprachige Query fuer Retrieval).
        Bei Fehlern bleibt der Originaltext erhalten.
        """
        guess = self.guess(question)
        if guess.code == "de":
            return guess, question
        return guess, self.to_german(question, guess.code)

    def render_answer(self, answer_de: str, target_lang: str) -> str:
        """
        Uebersetzt eine deutschsprachige Basisantwort in die Ziel-Language, sofern sinnvoll.
        Bei fehlender Erkennung bleibt Deutsch bestehen.
        """
        if target_lang == "de":
            return answer_de
        return self.from_german(answer_de, target_lang)
