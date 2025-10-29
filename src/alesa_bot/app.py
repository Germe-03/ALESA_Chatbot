from __future__ import annotations

"""
CLI Entry: nutzt die zentrale Factory (Separation of Concerns)
und startet die UI‑Laufschleife mit einem AppController.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

from src.alesa_bot.ui.cli import CliAdapter
from src.alesa_bot.runtime.runner import run_loop
from src.alesa_bot.runtime.factory import build_core, new_controller


def _build_dependencies():
    core = build_core()
    controller = new_controller(core)
    ui = CliAdapter()
    return controller, ui


def main() -> None:
    try:
        controller, ui = _build_dependencies()
    except Exception as e:
        print(f"Fehler bei Konfiguration/Initialisierung: {e}")
        return
    run_loop(controller, ui)


if __name__ == "__main__":
    main()

