from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")




import re

from src.alesa_bot.settings import load_config
from src.alesa_bot.retrieval.indexer import FileIndexer
from src.alesa_bot.retrieval.embeddings import EmbeddingEncoder
from src.alesa_bot.retrieval.hybrid import HybridRetriever
from src.alesa_bot.llm.vertex import VertexLLM
from src.alesa_bot.services.qa_service import QAService


def _banner() -> None:
    print("===" + " ALESA Chatbot ".center(50, "=") + "===")
    print("Tippe deine Nachricht und drücke [Enter].")
    print("Mit 'exit', 'quit' oder 'stop' beendest du den Chat.\n")


def read_user_input() -> str:
    """Entfernt versehentlich mitkopierte Prompt-Reste wie '👤 Du:' und harte Umbrüche."""
    raw = input("👤 Du: ")
    txt = re.sub(r'^(?:\s*👤\s*Du:\s*)+', '', raw)   # führende "👤 Du:"-Sequenzen entfernen
    txt = re.sub(r'[\r\n]+', ' ', txt)               # harte Zeilenumbrüche neutralisieren
    return txt.strip()


def main() -> None:
    # 1) Konfiguration laden (.env wird dabei gezogen)
    try:
        cfg = load_config()
    except Exception as e:
        print(f"⚠️  Konfiguration fehlgeschlagen: {e}")
        return

    # 2) Dateindex (TXT/MD/PDF) einmalig aufbauen
    indexer = FileIndexer(
        roots=[cfg.paths.data_processed_raw, cfg.paths.data_processed, cfg.paths.data_root],
        max_mb=cfg.retrieval.max_mb,
        max_pdf_pages=cfg.retrieval.max_pdf_pages,
    )
    indexer.build()

    # 3) Retriever (Hybrid: Embeddings + Lexikalisch via Chunks)
    encoder = EmbeddingEncoder(
        project=cfg.vertex.project,
        location=cfg.vertex.embed_location  # z. B. "us-central1" (regional!)
    )
    retriever = HybridRetriever(
        indexer=indexer,
        encoder=encoder,
        time_limit_sec=cfg.retrieval.time_limit_sec,
        chunk_size=800,
        overlap=200,
    )

    # 4) LLM-Adapter (VertexAI Gemini)
    llm = VertexLLM(
        project=cfg.vertex.project,
        location=cfg.vertex.location,
        model_name=cfg.vertex.model,
        creds_path=cfg.vertex.creds_path,
    )

    # 5) QA-Service (Orchestrierung: Retrieve → Prompt → Generate)
    service = QAService(
        retriever=retriever,
        llm=llm,
        system_prompt=cfg.system_prompt,
        query_expand=True,  # toleranter gegenüber unspezifischen Fragen
    )

    # 6) CLI-Loop
    _banner()
    while True:
        try:
            q = read_user_input()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Chat beendet.")
            break

        if q.lower() in {"exit", "quit", "stop"}:
            print("👋 Chat beendet.")
            break
        if not q:
            continue

        try:
            print("⏳ Verarbeite Anfrage …", flush=True)
            answer, cites = service.ask(q)
            print("✅ Fertig\n", flush=True)
        except ValueError as ve:
            # z.B. Guardrail „keine Quellen gefunden“
            print(f"🤖 ALESA: {ve}\n")
            continue
        except Exception as e:
            print(f"⚠️  Unerwarteter Fehler: {e}\n")
            continue

        print("🤖 ALESA:", answer, "\n")
        if cites:
            print("Quellen:\n" + "\n".join(cites) + "\n")


if __name__ == "__main__":
    main()
