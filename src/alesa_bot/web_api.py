from __future__ import annotations
import os
from pathlib import Path
from fastapi import FastAPI, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles
import uuid

from src.alesa_bot.runtime.factory import build_core, new_controller
from src.alesa_bot.services.qa_service import QAService
from src.alesa_bot.runtime.controller import AppController

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

# ------------------------------------------------------------
# Session controllers for full dialog (incl. orders)
# ------------------------------------------------------------
SESSIONS: dict[str, AppController] = {}

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
