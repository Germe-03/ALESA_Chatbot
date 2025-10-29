from __future__ import annotations
import os
from pathlib import Path
from fastapi import FastAPI, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles
import uuid

from src.alesa_bot.settings import load_config
from src.alesa_bot.retrieval.indexer import FileIndexer
from src.alesa_bot.retrieval.embeddings import EmbeddingEncoder
from src.alesa_bot.retrieval.hybrid import HybridRetriever
from src.alesa_bot.retrieval.vectorstore import ChromaStore, VSConfig
from src.alesa_bot.retrieval.boosted import BoostedRetriever
from src.alesa_bot.llm.vertex import VertexLLM
from src.alesa_bot.services.qa_service import QAService
from src.alesa_bot.services.order_repo import OrderRepo, OrderRecord
from src.alesa_bot.runtime.controller import AppController
from src.alesa_bot.assistant.preorder_gate import PreOrderGate
from src.alesa_bot.services.order_flow import OrderFlow
from src.alesa_bot.services.rma_flow import RMAFlow
from src.alesa_bot.services.rma_repo import RMARepo
from src.alesa_bot.retrieval.tables import ProductTableStore

# ------------------------------------------------------------
# App & CORS Setup
# ------------------------------------------------------------
app = FastAPI(title="ALESA API", version="1.0.0")

allow_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allow_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Public-Ordner (für index.html & Assets)
# ------------------------------------------------------------
PUBLIC_DIR = Path("public").resolve()
if PUBLIC_DIR.exists():
    assets_dir = PUBLIC_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# ------------------------------------------------------------
# Initialisierung von Config, Indexer, Retriever, LLM und QA-Service
# ------------------------------------------------------------
cfg = load_config()

indexer = FileIndexer(
    roots=[cfg.paths.data_processed_raw, cfg.paths.data_processed, cfg.paths.data_root],
    max_mb=cfg.retrieval.max_mb,
    max_pdf_pages=cfg.retrieval.max_pdf_pages,
)
indexer.build()

# Strukturierte Produkttabellen indexieren (CSV + heuristische PDF-Zeilen)
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

encoder = EmbeddingEncoder(
    project=cfg.vertex.project,
    location=cfg.vertex.embed_location,  # z. B. us-central1 (Embeddings)
)

try:
    store = ChromaStore(
        cfg=VSConfig(root=cfg.paths.data_root / "vectorstore"),
        encoder=encoder,
    )
    store.build(indexer, size=800, overlap=200)
    class _VSAdapter:
        def search(self, query: str, top_k: int = 6):
            return store.query(query, top_k=top_k)
    retriever = BoostedRetriever(indexer=indexer, base=_VSAdapter(), time_limit_sec=cfg.retrieval.time_limit_sec)
except Exception:
    # Fallback to hybrid in-memory if chroma is unavailable
    retriever = BoostedRetriever(indexer=indexer, base=HybridRetriever(
        indexer=indexer,
        encoder=encoder,
        time_limit_sec=cfg.retrieval.time_limit_sec,
        chunk_size=800,
        overlap=200,
    ), time_limit_sec=cfg.retrieval.time_limit_sec)

llm = VertexLLM(
    project=cfg.vertex.project,
    location=cfg.vertex.location,        # z. B. europe-west6 (Generative)
    model_name=cfg.vertex.model,
    creds_path=cfg.vertex.creds_path,    # nur genutzt, wenn gesetzt
)

qa = QAService(
    retriever=retriever,
    llm=llm,
    system_prompt=cfg.system_prompt,
    query_expand=True,
    product_store=prod_store,
)

# ------------------------------------------------------------
# Orders repo
# ------------------------------------------------------------
orders_repo = OrderRepo(root=cfg.paths.data_root / "orders")
rm_repo = RMARepo(root=cfg.paths.data_root / "orders")

# ------------------------------------------------------------
# Session controllers for full dialog (incl. orders)
# ------------------------------------------------------------
SESSIONS: dict[str, AppController] = {}

def _new_controller() -> AppController:
    return AppController(
        qa_service=QAService(
            retriever=retriever,
            llm=llm,
            system_prompt=cfg.system_prompt,
            query_expand=True,
            product_store=prod_store,
        ),
        order_flow=OrderFlow(),
        pre_order_gate=PreOrderGate(),
        orders_repo=orders_repo,
        rma_flow=RMAFlow(),
        rma_repo=rm_repo,
    )

def _get_controller(sid: str) -> AppController:
    c = SESSIONS.get(sid)
    if c is None:
        c = _new_controller()
        SESSIONS[sid] = c
    return c

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(question: str = Form(...)):
    """Empfängt eine Frage vom Frontend und liefert Antwort + Quellen."""
    try:
        answer, cites = qa.ask(question)
        return {"answer": answer, "sources": cites}

    except ValueError as ve:
        # z. B. Guardrail: „Keine passenden Quellen gefunden.“
        return {"answer": f"[Hinweis] {str(ve)}", "sources": []}

    except Exception as e:
        # Unerwarteter Fehler → HTTP 500 mit generischer Meldung
        print("❌ ERROR in /ask:", e)
        raise HTTPException(status_code=500, detail="Serverfehler beim Beantworten der Frage.")


@app.get("/")
def home():
    """Liefert die HTML-Oberfläche (public/index.html)."""
    index_file = PUBLIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"detail": "Not Found", "hint": "Leg public/index.html an oder nutze /ask direkt per API."}


# ------------------------------------------------------------
# Employee Portal (Admin)
# ------------------------------------------------------------
@app.get("/admin")
def admin_home():
    admin_index = PUBLIC_DIR / "admin" / "index.html"
    if admin_index.exists():
        return FileResponse(str(admin_index))
    return {"detail": "Not Found", "hint": "Leg public/admin/index.html an."}


@app.get("/admin/orders")
def list_orders():
    return {"orders": orders_repo.list(limit=500)}


@app.post("/admin/order")
def add_order(payload: dict = Body(...)):
    try:
        rec = OrderRecord(
            id=orders_repo.new_id(),
            created_at=orders_repo.now_iso(),
            customer=payload.get("customer", {}),
            items=payload.get("items", []),
            comment=payload.get("comment"),
        )
        orders_repo.add(rec)
        return {"ok": True, "id": rec.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid order payload: {e}")


# ------------------------------------------------------------
# Chat endpoint with session state (Controller orchestration)
# ------------------------------------------------------------
@app.post("/chat")
def chat(session: str = Form(None), message: str = Form(...)):
    sid = session or uuid.uuid4().hex
    ctrl = _get_controller(sid)
    try:
        replies = ctrl.handle(message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")
    return {"session": sid, "responses": replies or []}
