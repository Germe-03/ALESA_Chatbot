# -*- coding: utf-8 -*-
import os
import sys
from typing import Tuple, List
from dotenv import load_dotenv
from vertexai import init as vertex_init
from vertexai.generative_models import GenerativeModel
from google.api_core.exceptions import NotFound, PermissionDenied

def _resolve_creds_path(creds: str) -> str:
    """Nimmt Datei- oder Ordnerpfad. Liefert absoluten Pfad zur JSON-Datei."""
    if not creds:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS fehlt in .env")

    creds = os.path.expanduser(creds)
    creds = os.path.abspath(creds)

    if os.path.isfile(creds):
        return creds

    if os.path.isdir(creds):
        # Alle JSON-Dateien im Ordner sammeln
        cand = [os.path.join(creds, f) for f in os.listdir(creds) if f.lower().endswith(".json")]
        if len(cand) == 1:
            return cand[0]
        if len(cand) == 0:
            raise RuntimeError(
                f"Im Ordner {creds} wurde keine *.json gefunden. "
                "Bitte den exakten Pfad zur Keydatei angeben."
            )
        raise RuntimeError(
            f"Im Ordner {creds} wurden mehrere *.json gefunden. "
            "Bitte genaue Datei in GOOGLE_APPLICATION_CREDENTIALS angeben."
        )

    raise RuntimeError(f"Pfad {creds} existiert nicht.")

def read_env() -> Tuple[str, str, str, str]:
    load_dotenv()
    project = os.getenv("GCP_PROJECT", "").strip()
    location = os.getenv("GCP_LOCATION", "global").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

    if not project:
        raise RuntimeError("GCP_PROJECT fehlt in .env")
    if not location:
        raise RuntimeError("GCP_LOCATION fehlt in .env")
    if not model:
        raise RuntimeError("GEMINI_MODEL fehlt in .env")

    creds = _resolve_creds_path(creds)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds  # gcloud libs nutzen das
    return project, location, model, creds

def print_vertex_header(project: str, location: str, model: str):
    print(f"[Vertex] project={project}  location={location}  model={model}", flush=True)

def run_chat(model_id: str) -> str:
    model = GenerativeModel(model_id)
    chat = model.start_chat(history=[])
    resp = chat.send_message("Wie funktionierst du, wie heisst du (coPilot, Chat GPT)?")
    return (resp.text or "").strip()

def main():
    try:
        project, location, model_id, _ = read_env()
        vertex_init(project=project, location=location)
        print_vertex_header(project, location, model_id)
        print("\n=== Antwort ===")
        print(run_chat(model_id))
        return

    except NotFound as e:
        print("[Warn] Publisher Model nicht gefunden oder kein Zugriff.\n→ versuche Fallbacks ...")
        fallbacks: List[Tuple[str, str]] = [
            ("gemini-2.0-flash", "location"),
            ("gemini-2.0-flash-001", "location"),
            ("gemini-2.5-flash", "global"),
            ("gemini-2.5-pro", "global"),
            ("gemini-2.0-flash", "us-central1"),
            ("gemini-2.0-flash", "europe-west1"),
        ]
        last_err = e
        for mdl, loc in fallbacks:
            try:
                print(f"[Try] model={mdl}  location={loc}")
                vertex_init(project=project, location=loc)
                print(run_chat(mdl))
                return
            except NotFound as nf:
                last_err = nf
                continue
        raise last_err

    except PermissionDenied:
        print("⚠️  Berechtigung fehlt. Prüfe Rollen: Vertex AI User, Service Usage Consumer, ggf. Storage Object Viewer.")
        raise
    except Exception as e:
        print("Unerwarteter Fehler:", e)
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
