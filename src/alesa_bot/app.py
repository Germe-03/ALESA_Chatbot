from __future__ import annotations

"""
Composition Root für ALESA (Start über diese Datei).

Diese Datei:
- lädt die Konfiguration
- initialisiert Index/Retrieval/LLM/Services
- steckt die Dialog-Orchestrierung zusammen
- übergibt alles an die UI-Laufschleife

Hinweis zu den Imports:
- Um den Namenskonflikt zwischen diesem Script `app.py` und einem gleichnamigen
  Package-Verzeichnis zu vermeiden, liegen Controller/Runner in einem Package
  namens `runtime` (statt `app`).
- Nächster Schritt: `controller.py` und `runner.py` unter
  `src/alesa_bot/runtime/` bereitstellen.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

from src.alesa_bot.settings import load_config
from src.alesa_bot.retrieval.indexer import FileIndexer
from src.alesa_bot.retrieval.embeddings import EmbeddingEncoder
from src.alesa_bot.retrieval.hybrid import HybridRetriever
from src.alesa_bot.llm.vertex import VertexLLM
from src.alesa_bot.services.qa_service import QAService
from src.alesa_bot.services.order_flow import OrderFlow

# Intent-/Gate-Logik
from src.alesa_bot.assistant.preorder_gate import PreOrderGate

# ⚠️ WICHTIG: Controller/Runner kommen aus dem Package `runtime` (nicht `app`)
from src.alesa_bot.ui.cli import CliAdapter
from src.alesa_bot.runtime.controller import AppController   # <— neu: Pfad geändert
from src.alesa_bot.runtime.runner import run_loop            # <— neu: Pfad geändert


def _build_dependencies():
    """
    Setzt alle Kernabhängigkeiten zusammen und liefert Controller + UI zurück.
    """
    cfg = load_config()

    # 1) Dateindex aufbauen
    indexer = FileIndexer(
        roots=[cfg.paths.data_processed_raw, cfg.paths.data_processed, cfg.paths.data_root],
        max_mb=cfg.retrieval.max_mb,
        max_pdf_pages=cfg.retrieval.max_pdf_pages,
    )
    indexer.build()

    # 2) Retriever (Embeddings + Hybrid Ranking)
    encoder = EmbeddingEncoder(
        project=cfg.vertex.project,
        location=cfg.vertex.embed_location
    )
    retriever = HybridRetriever(
        indexer=indexer,
        encoder=encoder,
        time_limit_sec=cfg.retrieval.time_limit_sec,
        chunk_size=800,
        overlap=200,
    )

    # 3) LLM
    llm = VertexLLM(
        project=cfg.vertex.project,
        location=cfg.vertex.location,
        model_name=cfg.vertex.model,
        creds_path=cfg.vertex.creds_path,
    )

    # 4) QA-Service
    qa_service = QAService(
        retriever=retriever,
        llm=llm,
        system_prompt=cfg.system_prompt,
        query_expand=True,
    )

    # 5) Order & Gate
    order_flow = OrderFlow()
    pre_gate = PreOrderGate()

    # 6) UI-Port (CLI)
    ui = CliAdapter()

    # 7) Controller (Dialog-Orchestrierung)
    controller = AppController(
        qa_service=qa_service,
        order_flow=order_flow,
        pre_order_gate=pre_gate,
        system_banner=(
            "===" + " ALESA Chatbot ".center(50, "=") + "===\n"
            "👋 Hallo! Ich bin ALESA, dein virtueller KI-Assistent.\n"
            "Ich unterstütze dich bei allgemeinen Fragen, bei Produktempfehlungen\n"
            "und – wenn du möchtest – auch direkt beim Bestellen.\n"
            "Sag einfach z. B. „Ich würde gerne bestellen“, um den Bestell-Assistenten zu starten.\n\n"
            "Tippe deine Nachricht und drücke [Enter].\n"
            "Mit 'exit', 'quit' oder 'stop' beendest du den Chat.\n"
        )
    )

    return controller, ui


def main() -> None:
    try:
        controller, ui = _build_dependencies()
    except Exception as e:
        print(f"⚠️  Konfiguration/Initialisierung fehlgeschlagen: {e}")
        return

    run_loop(controller, ui)


if __name__ == "__main__":
    main()
