from __future__ import annotations
"""
App Runner: entkoppelte, testbare Hauptschleife für die ALESA-CLI.

Verantwortung:
- Startbanner anzeigen
- Optionalen Starttext des Controllers ausgeben
- Eingaben lesen, Exit-Kommandos behandeln
- Eingaben an den Controller delegieren
- Antworten über die UI ausgeben

Wichtig:
- Keine Geschäftslogik und kein direkter Terminal-Zugriff (läuft über UiPort).
"""

from typing import Protocol, List
from src.alesa_bot.ui.cli import UiPort


class ControllerPort(Protocol):
    """
    Minimaler Controller-Vertrag, den der Runner benötigt.
    Der konkrete Controller steckt die Dialog-Orchestrierung zusammen.
    """
    @property
    def system_banner(self) -> str:
        ...

    def on_start(self) -> str | None:
        """Optionaler Initialtext des Bots (z. B. Hinweise)."""
        ...

    def handle(self, user_text: str) -> List[str]:
        """
        Verarbeitet eine Nutzereingabe und liefert 0..n Bot-Nachrichten zurück,
        die der Runner über die UI ausgibt.
        """
        ...


_EXIT_CMDS = {"exit", "quit", "stop"}


def run_loop(controller: ControllerPort, ui: UiPort) -> None:
    """
    Führt die Interaktionsschleife aus. Ist UI-agnostisch dank UiPort.
    """
    # 1) Banner
    banner = getattr(controller, "system_banner", "") or ""
    if banner:
        ui.show_banner(banner)

    # 2) Optionaler Initialtext
    try:
        initial = controller.on_start()
    except Exception as e:
        ui.show_error(f"Fehler beim Initialisieren des Controllers: {e}")
        initial = None
    if initial:
        ui.show(initial)

    # 3) Hauptschleife
    while True:
        try:
            user_text = ui.read()
        except (EOFError, KeyboardInterrupt):
            ui.show("👋 Chat beendet.")
            break

        # Exit-Kommandos global abfangen
        if (user_text or "").strip().lower() in _EXIT_CMDS:
            ui.show("👋 Chat beendet.")
            break

        # Leere Eingaben erlauben: Controller entscheidet, ob/was passiert
        try:
            responses = controller.handle(user_text)
        except Exception as e:
            ui.show_error(f"Unerwarteter Fehler: {e}")
            continue

        if not responses:
            # Controller hat bewusst nichts zu sagen.
            continue

        for msg in responses:
            if msg is None:
                continue
            ui.show(str(msg))
