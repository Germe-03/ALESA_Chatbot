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
from src.alesa_bot.retrieval.vectorstore import ChromaStore, VSConfig
from src.alesa_bot.retrieval.boosted import BoostedRetriever
from src.alesa_bot.retrieval.tables import ProductTableStore
from src.alesa_bot.llm.vertex import VertexLLM
from src.alesa_bot.services.qa_service import QAService
from src.alesa_bot.services.order_flow import OrderFlow
from src.alesa_bot.services.order_repo import OrderRepo
from src.alesa_bot.services.rma_flow import RMAFlow
from src.alesa_bot.services.rma_repo import RMARepo

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

    # 1b) Strukturierte Produkttabellen laden (CSV + heuristische PDF-Zeilen)
    prod_store = ProductTableStore()
    try:
        csv_dir1 = cfg.paths.data_root / "products"
        csv_dir2 = cfg.paths.data_processed / "products"
        csv_files = []
        for d in [csv_dir1, csv_dir2]:
            if d.exists():
                csv_files.extend([p for p in d.rglob("*.csv")])
        prod_store.ingest_csv(csv_files)
        prod_store.ingest_from_indexer(indexer)
    except Exception:
        pass

    # 2) Retriever (Embeddings + Hybrid Ranking)
    encoder = EmbeddingEncoder(
        project=cfg.vertex.project,
        location=cfg.vertex.embed_location
    )
    # 2b) Vectorstore (Chroma) as primary retriever
    try:
        store = ChromaStore(
            cfg=VSConfig(root=cfg.paths.data_root / "vectorstore"),
            encoder=encoder,
        )
        added, _ = store.build(indexer, size=800, overlap=200)
        base = type("_VSAdapter", (), {
            "search": lambda self, q, top_k=6: store.query(q, top_k=top_k)
        })()
        retriever = BoostedRetriever(indexer=indexer, base=base, time_limit_sec=cfg.retrieval.time_limit_sec)
    except Exception:
        # Fallback to in-memory hybrid retriever if chroma not available
        base = HybridRetriever(
            indexer=indexer,
            encoder=encoder,
            time_limit_sec=cfg.retrieval.time_limit_sec,
            chunk_size=800,
            overlap=200,
        )
        retriever = BoostedRetriever(indexer=indexer, base=base, time_limit_sec=cfg.retrieval.time_limit_sec)

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
        product_store=prod_store,
    )

    # 5) Order & Gate
    order_flow = OrderFlow()
    orders_repo = OrderRepo(root=cfg.paths.data_root / "orders")
    rma_flow = RMAFlow()
    rma_repo = RMARepo(root=cfg.paths.data_root / "orders")
    pre_gate = PreOrderGate()

    # 6) UI-Port (CLI)
    ui = CliAdapter()

    # 7) Controller (Dialog-Orchestrierung)
    controller = AppController(
        qa_service=qa_service,
        order_flow=order_flow,
        pre_order_gate=pre_gate,
        orders_repo=orders_repo,
        rma_flow=rma_flow,
        rma_repo=rma_repo,
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
