from __future__ import annotations
import os
from pathlib import Path
from fastapi import FastAPI, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles
import uuid

from src.alesa_bot.runtime.factory import build_core, new_controller
from src.alesa_bot.services.order_repo import OrderRecord
from src.alesa_bot.services.qa_service import QAService
from src.alesa_bot.runtime.controller import AppController
from src.alesa_bot.analytics.query_clustering import build_clusters_cache, get_cached_clusters
from uuid import uuid4

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

core = build_core()
qa = QAService(
    retriever=core.retriever,
    llm=core.llm,
    system_prompt=core.cfg.system_prompt,
    query_expand=True,
    product_store=core.prod_store,
)
# Cluster einmal beim Start berechnen (Admin-Analytics)
clusters_snapshot = build_clusters_cache()

# ------------------------------------------------------------
# Session controllers for full dialog (incl. orders)
# ------------------------------------------------------------
SESSIONS: dict[str, AppController] = {}
# Seed per Server-Start, um Chat-Session-IDs nach Restart neu zu machen
SERVER_SEED = uuid4().hex[:8]

def _new_controller() -> AppController:
    return new_controller(core)

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
    return {"detail": "Not Found", "hint": "Leg public/index.html an oder nutze /chat per API."}


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
    return {"orders": core.orders_repo.list(limit=500)}


@app.post("/admin/order")
def add_order(payload: dict = Body(...)):
    try:
        rec = OrderRecord(
            id=core.orders_repo.new_id(),
            created_at=core.orders_repo.now_iso(),
            customer=payload.get("customer", {}),
            items=payload.get("items", []),
            comment=payload.get("comment"),
        )
        core.orders_repo.add(rec)
        return {"ok": True, "id": rec.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid order payload: {e}")


@app.get("/admin/chats")
def list_chats():
    if core.logger is None:
        return {"chats": []}
    return {"chats": core.logger.list_sessions(limit=300)}


@app.get("/admin/chats/{session_id}")
def get_chat(session_id: str):
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session id")
    if core.logger is None:
        return {"messages": []}
    return {"messages": core.logger.list_messages(session_id=session_id, limit=800)}


@app.get("/admin/clusters")
def list_clusters(n_clusters: int = 8):
    try:
        # Auf Anfrage andere k berechnen; sonst Start-Snapshot nutzen
        if n_clusters != 8:
            return build_clusters_cache(n_clusters=n_clusters)
        return get_cached_clusters(n_clusters=n_clusters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {e}")


# ------------------------------------------------------------
# Chat endpoint with session state (Controller orchestration)
# ------------------------------------------------------------
@app.post("/chat")
def chat(session: str = Form(None), message: str = Form(...)):
    sid_raw = session or uuid.uuid4().hex
    # Prefix mit SERVER_SEED erzwingt neue Chat-ID pro Server-Start, auch wenn der Client die gleiche Basis-ID sendet
    sid = f"{SERVER_SEED}-{sid_raw}"
    ctrl = _get_controller(sid)
    try:
        # log user message
        try:
            core.logger.log_interaction(sid, role="user", message=message)
        except Exception:
            pass
        replies = ctrl.handle(message, session_id=sid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")
    # log bot replies with optional sources from QA
    try:
        sources = getattr(ctrl.qa_service, "last_sources", None)
        keywords = getattr(ctrl.qa_service, "last_keywords", None)
        for r in replies or []:
            core.logger.log_interaction(sid, role="bot", message=str(r), sources=sources, keywords=keywords)
    except Exception:
        pass
    return {"session": sid, "responses": replies or []}
