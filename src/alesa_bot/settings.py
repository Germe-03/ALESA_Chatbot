# ===================== FILE: src/alesa_bot/settings.py =====================
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv, dotenv_values

@dataclass(frozen=True)
class VertexConfig:
    # Chat/Gemini
    project: str
    location: str
    model: str
    creds_path: str
    # Embeddings (regional!)
    embed_location: str = "us-central1"
    embed_model: str = "text-embedding-004"

@dataclass(frozen=True)
class RetrievalConfig:
    max_mb: int = 15
    max_pdf_pages: int = 15
    time_limit_sec: int = 4

@dataclass(frozen=True)
class Paths:
    root: Path
    data_root: Path
    data_processed: Path
    data_processed_raw: Path

@dataclass(frozen=True)
class Config:
    vertex: VertexConfig
    retrieval: RetrievalConfig
    paths: Paths
    system_prompt: str

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _load_env_into_os() -> None:
    # Reihenfolge: expliziter Pfad → Projektwurzel → CWD
    candidates = []
    env_override = os.environ.get("ALESA_ENV_PATH")
    if env_override:
        candidates.append(Path(env_override))
    candidates += [_project_root() / ".env", Path.cwd() / ".env"]
    for p in candidates:
        if p.exists():
            load_dotenv(p, override=False)
            dotenv_values(p)  # prüft Datei grob
            return

def load_config() -> Config:
    _load_env_into_os()

    required = ["GCP_PROJECT", "GCP_LOCATION", "GEMINI_MODEL",
                "GOOGLE_APPLICATION_CREDENTIALS", "SYSTEM_PROMPT"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Fehlende .env Variablen: {', '.join(missing)}")

    root = _project_root()
    paths = Paths(
        root=root,
        data_root=root / "data",
        data_processed=root / "data" / "processed",
        data_processed_raw=root / "data" / "processed" / "raw",
    )

    vertex = VertexConfig(
        project=os.environ["GCP_PROJECT"],
        location=os.environ.get("GCP_LOCATION", "global"),
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        creds_path=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        embed_location=os.environ.get("EMBEDDING_LOCATION", "us-central1"),
        embed_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-004"),
    )

    retrieval = RetrievalConfig()
    system_prompt = os.environ.get(
        "SYSTEM_PROMPT", "Du bist ALESA, ein hilfreicher und freundlicher KI-Assistent."
    )
    return Config(vertex=vertex, retrieval=retrieval, paths=paths, system_prompt=system_prompt)
