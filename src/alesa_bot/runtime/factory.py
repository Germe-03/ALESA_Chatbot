from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
from src.alesa_bot.services.order_service import OrderService
from src.alesa_bot.services.rma_service import RMAService
from src.alesa_bot.assistant.preorder_gate import PreOrderGate
from src.alesa_bot.runtime.controller import AppController


@dataclass(frozen=True)
class CoreContext:
    cfg: any
    indexer: FileIndexer
    encoder: EmbeddingEncoder
    retriever: any
    llm: VertexLLM
    prod_store: ProductTableStore
    orders_repo: OrderRepo
    rma_repo: RMARepo
    order_service: OrderService | None = None
    rma_service: RMAService | None = None


def _build_index_and_tables(cfg) -> tuple[FileIndexer, ProductTableStore]:
    indexer = FileIndexer(
        roots=[cfg.paths.data_processed_raw, cfg.paths.data_processed, cfg.paths.data_root],
        max_mb=cfg.retrieval.max_mb,
        max_pdf_pages=cfg.retrieval.max_pdf_pages,
    )
    indexer.build()

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
    return indexer, prod_store


def _build_retriever(cfg, indexer) -> tuple[EmbeddingEncoder, any]:
    encoder = EmbeddingEncoder(
        project=cfg.vertex.project,
        location=cfg.vertex.embed_location,
    )
    try:
        store = ChromaStore(cfg=VSConfig(root=cfg.paths.data_root / "vectorstore"), encoder=encoder)
        store.build(indexer, size=800, overlap=200)

        class _VSAdapter:
            def search(self, query: str, top_k: int = 6):
                return store.query(query, top_k=top_k)

        base = _VSAdapter()
    except Exception:
        base = HybridRetriever(
            indexer=indexer,
            encoder=encoder,
            time_limit_sec=cfg.retrieval.time_limit_sec,
            chunk_size=800,
            overlap=200,
        )

    retriever = BoostedRetriever(indexer=indexer, base=base, time_limit_sec=cfg.retrieval.time_limit_sec)
    return encoder, retriever


def _build_llm(cfg) -> VertexLLM:
    return VertexLLM(
        project=cfg.vertex.project,
        location=cfg.vertex.location,
        model_name=cfg.vertex.model,
        creds_path=cfg.vertex.creds_path,
    )


def build_core() -> CoreContext:
    cfg = load_config()
    indexer, prod_store = _build_index_and_tables(cfg)
    encoder, retriever = _build_retriever(cfg, indexer)
    llm = _build_llm(cfg)
    orders_repo = OrderRepo(root=cfg.paths.data_root / "orders")
    rma_repo = RMARepo(root=cfg.paths.data_root / "orders")
    order_service = OrderService(orders_repo)
    rma_service = RMAService(rma_repo)
    return CoreContext(cfg, indexer, encoder, retriever, llm, prod_store, orders_repo, rma_repo, order_service, rma_service)


def new_controller(core: CoreContext) -> AppController:
    qa = QAService(
        retriever=core.retriever,
        llm=core.llm,
        system_prompt=core.cfg.system_prompt,
        query_expand=True,
        product_store=core.prod_store,
    )
    return AppController(
        qa_service=qa,
        order_flow=OrderFlow(),
        pre_order_gate=PreOrderGate(),
        orders_repo=core.orders_repo,
        order_service=core.order_service,
        rma_flow=RMAFlow(),
        rma_repo=core.rma_repo,
        rma_service=core.rma_service,
        system_banner=banner_text(),
    )


def banner_text() -> str:
    return (
        "===" + " ALESA Chatbot ".center(50, "=") + "===\n"
        "👋 Hallo! Ich bin ALESA, dein virtueller KI-Assistent.\n"
        "Ich unterstütze dich bei allgemeinen Fragen, bei Produktempfehlungen\n"
        "und — wenn du möchtest — auch direkt beim Bestellen.\n"
        "Sag einfach z. B. ‘Ich würde gerne bestellen’, um den Bestell-Assistenten zu starten.\n\n"
        "Tippe deine Nachricht und drücke [Enter].\n"
        "Mit 'exit', 'quit' oder 'stop' beendest du den Chat.\n"
    )
