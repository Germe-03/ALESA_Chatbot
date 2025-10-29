from __future__ import annotations
import os
from pathlib import Path
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles

from src.alesa_bot.settings import load_config
from src.alesa_bot.retrieval.indexer import FileIndexer
from src.alesa_bot.retrieval.embeddings import EmbeddingEncoder
from src.alesa_bot.retrieval.hybrid import HybridRetriever
from src.alesa_bot.retrieval.vectorstore import ChromaStore, VSConfig
from src.alesa_bot.llm.vertex import VertexLLM
from src.alesa_bot.services.qa_service import QAService

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
    retriever = _VSAdapter()
except Exception:
    # Fallback to hybrid in-memory if chroma is unavailable
    retriever = HybridRetriever(
        indexer=indexer,
        encoder=encoder,
        time_limit_sec=cfg.retrieval.time_limit_sec,
        chunk_size=800,
        overlap=200,
    )

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
)

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
