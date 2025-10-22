from __future__ import annotations
"""
CLI-Adapter für die ALESA-App.

Zweck:
- Kapselt alle Terminal-Ein-/Ausgaben (print/input) hinter einem schmalen UI-Port.
- Erlaubt alternative UIs (z. B. Web, GUI) ohne Änderung der Dialoglogik.

Verwendung:
- Der Runner ruft `ui.show_banner(...)` einmal zu Beginn auf.
- Für jede Nutzereingabe ruft der Runner `ui.read()` auf und schreibt Antworten mit `ui.show(...)`.
- Fehler-/Warnhinweise gehen über `ui.show_error(...)`.

Hinweis:
- Die Eingabe wird ähnlich wie im früheren app.py bereinigt (entfernt "👤 Du:"-Präfix und harte Umbrüche).
"""

from typing import Protocol, Callable
import re
import sys


class UiPort(Protocol):
    """Abstrakte UI-Schnittstelle, die vom Runner/Controller genutzt wird."""

    def show_banner(self, text: str) -> None:
        """Initiale Begrüßung / Systembanner ausgeben."""
        ...

    def show(self, text: str) -> None:
        """Normale Bot-Ausgaben anzeigen."""
        ...

    def show_error(self, text: str) -> None:
        """Fehlermeldungen/Warnings anzeigen."""
        ...

    def read(self) -> str:
        """Eine Nutzereingabe lesen (bereinigt zurückgeben)."""
        ...


class CliAdapter(UiPort):
    """
    Einfache CLI-Implementierung von UiPort.

    - Standardmäßig nutzt sie `input()`/`print()`.
    - Für Tests können Ein-/Ausgabefunktionen injiziert werden.
    """

    def __init__(
        self,
        input_fn: Callable[[str], str] | None = None,
        output_fn: Callable[[str], None] | None = None,
        error_fn: Callable[[str], None] | None = None,
        user_prompt: str = "👤 Du: ",
        bot_prefix: str = "🤖 ALESA: ",
    ) -> None:
        self._input = input_fn or input
        self._out = output_fn or (lambda s: print(s, flush=True))
        self._err = error_fn or (lambda s: print(s, file=sys.stderr, flush=True))
        self._user_prompt = user_prompt
        self._bot_prefix = bot_prefix

        # kompiliere Regexe einmal
        self._rx_prefix = re.compile(r'^(?:\s*👤\s*Du:\s*)+', flags=re.IGNORECASE)
        self._rx_newlines = re.compile(r'[\r\n]+')

    # -------- UiPort API --------

    def show_banner(self, text: str) -> None:
        self._out(text.rstrip("\n"))

    def show(self, text: str) -> None:
        """Bot-Antworten normiert ausgeben (mit Prefix)."""
        # Wir fügen den Prefix nur hinzu, wenn er nicht bereits enthalten ist (z. B. bei mehrzeiligem Block).
        if text.startswith(self._bot_prefix):
            self._out(text)
        else:
            self._out(f"{self._bot_prefix}{text}")

    def show_error(self, text: str) -> None:
        self._err(text.rstrip("\n"))

    def read(self) -> str:
        """Nutzereingabe lesen und bereinigen."""
        raw = self._input(self._user_prompt)
        return self._sanitize_user_input(raw)

    # -------- Hilfen --------

    def _sanitize_user_input(self, raw: str) -> str:
        """
        Entfernt versehentlich mitkopierte Prompt-Reste wie '👤 Du:' und harte Umbrüche.
        Entspricht der bisherigen Logik in app.py, aber gekapselt.
        """
        if raw is None:
            return ""
        txt = self._rx_prefix.sub("", raw)
        txt = self._rx_newlines.sub(" ", txt)
        return txt.strip()
