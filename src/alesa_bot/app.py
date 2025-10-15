# ===================== FILE: src/alesa_bot/app.py =====================
from __future__ import annotations
from src.alesa_bot.settings import load_config
from src.alesa_bot.retrieval.indexer import FileIndexer
from src.alesa_bot.retrieval.search import SimpleRetriever
from src.alesa_bot.llm.vertex import VertexLLM
from src.alesa_bot.services.qa_service import QAService


def main() -> None:
    cfg = load_config()

    indexer = FileIndexer(
        roots=[cfg.paths.data_processed_raw, cfg.paths.data_processed, cfg.paths.data_root],
        max_mb=cfg.retrieval.max_mb,
        max_pdf_pages=cfg.retrieval.max_pdf_pages,
    )
    indexer.build()

    retriever = SimpleRetriever(indexer=indexer, time_limit_sec=cfg.retrieval.time_limit_sec)
    llm = VertexLLM(cfg.vertex.project, cfg.vertex.location, cfg.vertex.model, cfg.vertex.creds_path)
    service = QAService(retriever=retriever, llm=llm, system_prompt=cfg.system_prompt)

    print("===" + " ALESA Chatbot ".center(50, "=") + "===")
    print("Tippe deine Nachricht und drücke [Enter].")
    print("Mit 'exit', 'quit' oder 'stop' beendest du den Chat.\n")

    while True:
        try:
            q = input("👤 Du: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Chat beendet.")
            break
        if q.lower() in {"exit", "quit", "stop"}:
            print("👋 Chat beendet.")
            break
        if not q:
            continue

        answer, cites = service.ask(q)
        print("🤖 ALESA:", answer, "\n")
        if cites:
            print("Quellen:\n" + "\n".join(cites) + "\n")


if __name__ == "__main__":
    main()

















