from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

import re
import re as _re

from src.alesa_bot.settings import load_config
from src.alesa_bot.retrieval.indexer import FileIndexer
from src.alesa_bot.retrieval.embeddings import EmbeddingEncoder
from src.alesa_bot.retrieval.hybrid import HybridRetriever
from src.alesa_bot.llm.vertex import VertexLLM
from src.alesa_bot.services.qa_service import QAService
from src.alesa_bot.services.order_flow import OrderFlow


def _banner() -> None:
    print("===" + " ALESA Chatbot ".center(50, "=") + "===")
    print(
        "👋 Hallo! Ich bin ALESA, dein virtueller KI-Assistent.\n"
        "Ich unterstütze dich bei allgemeinen Fragen, bei Produktempfehlungen\n"
        "und – wenn du möchtest – auch direkt beim Bestellen.\n"
        "Sag einfach z. B. „Ich würde gerne bestellen“, um den Bestell-Assistenten zu starten.\n"
    )
    print("Tippe deine Nachricht und drücke [Enter].")
    print("Mit 'exit', 'quit' oder 'stop' beendest du den Chat.\n")


def read_user_input() -> str:
    """Entfernt versehentlich mitkopierte Prompt-Reste wie '👤 Du:' und harte Umbrüche."""
    raw = input("👤 Du: ")
    txt = re.sub(r'^(?:\s*👤\s*Du:\s*)+', '', raw)
    txt = re.sub(r'[\r\n]+', ' ', txt)
    return txt.strip()


# ---------- Intent-Helfer ----------

_INTERROGATIVES = (
    "was", "welche", "welcher", "welches", "wie", "wo", "wann",
    "wer", "wieso", "weshalb", "warum", "kann", "darf", "soll",
)

_PURCHASE_REGEXES = [
    _re.compile(r"\bich\s+(?:möchte|will|würde(?:\s+gern(?:e)?)?)\b.*\bbestellen\b", _re.IGNORECASE),
    _re.compile(r"\bkann\s+(?:ich|man)\b.*\bbestellen\b", _re.IGNORECASE),
    _re.compile(r"\bbitte\b.*\bbestellen\b", _re.IGNORECASE),
    _re.compile(r"\b(?:ich\s+)?bestelle\b.*", _re.IGNORECASE),
    _re.compile(r"\bmöchte\b.*\bbestellen\b", _re.IGNORECASE),
]

def is_question(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if "?" in t:
        return True
    start = t.split(maxsplit=1)[0].lower()
    return start in _INTERROGATIVES

def is_strong_purchase_intent(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if is_question(t):
        return False
    return any(rx.search(t) for rx in _PURCHASE_REGEXES)


# --- PreOrderGate (unverändert bis auf suppress-Flag) ---
class PreOrderGate:
    def __init__(self) -> None:
        self.await_choice = False
        self.product_hint = ""
        self.cached_user_query = ""
        self.mode = "neutral"              # "neutral" | "advice"
        self.suppress_next_gate = False

    @staticmethod
    def _extract_product_hint(text: str) -> str:
        low = text.lower()
        if "nutex" in low:
            return "Nutex"
        return ""

    def start(self, user_text: str) -> str:
        self.await_choice = True
        self.cached_user_query = user_text
        self.product_hint = self._extract_product_hint(user_text)
        prod = f" **{self.product_hint}**" if self.product_hint else ""
        return (
            f"Möchtest du{prod} **direkt bestellen** oder zuerst eine **Beratung/Produktempfehlung**?\n"
            "Bitte antworte mit **bestellen** oder **beratung**."
        )

    def active(self) -> bool:
        return self.await_choice

    def handle_choice(self, user_text: str) -> str:
        low = user_text.strip().lower()
        if "bestellen" in low:
            self.await_choice = False
            self.mode = "neutral"
            return "go_order"
        if "beratung" in low or "empfehlung" in low:
            self.await_choice = False
            self.mode = "advice"
            self.suppress_next_gate = True
            return "go_qa"
        return "repeat"

    def should_prompt_gate(self, user_text: str) -> bool:
        if is_question(user_text):
            return False
        return is_strong_purchase_intent(user_text)


def main() -> None:
    # 1) Konfiguration laden
    try:
        cfg = load_config()
    except Exception as e:
        print(f"⚠️  Konfiguration fehlgeschlagen: {e}")
        return

    # 2) Dateindex
    indexer = FileIndexer(
        roots=[cfg.paths.data_processed_raw, cfg.paths.data_processed, cfg.paths.data_root],
        max_mb=cfg.retrieval.max_mb,
        max_pdf_pages=cfg.retrieval.max_pdf_pages,
    )
    indexer.build()

    # 3) Retriever
    encoder = EmbeddingEncoder(
        project=cfg.vertex.project,
        location=cfg.vertex.embed_location
    )
    retriever = HybridRetriever(
        indexer=indexer,
        encoder=encoder,
        time_limit_sec=cfg.retrieval.time_limit_sec,
        chunk_size=800,
        overlap=200,
    )

    # 4) LLM
    llm = VertexLLM(
        project=cfg.vertex.project,
        location=cfg.vertex.location,
        model_name=cfg.vertex.model,
        creds_path=cfg.vertex.creds_path,
    )

    # 5) QA-Service
    service = QAService(
        retriever=retriever,
        llm=llm,
        system_prompt=cfg.system_prompt,
        query_expand=True,
    )

    # 6) Order + Gate
    order_flow = OrderFlow()
    pre_gate = PreOrderGate()

    # 7) Loop
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

        # --- WICHTIG: Leere Eingaben im aktiven Bestellflow NICHT überspringen ---
        if not q:
            if order_flow.is_active():
                reply = order_flow.handle("")
                print("🤖 ALESA:", reply, "\n")
                continue
            else:
                continue

        # A) OrderFlow aktiv → Vorrang
        if order_flow.is_active():
            reply = order_flow.handle(q)
            print("🤖 ALESA:", reply, "\n")
            continue

        # B) Gate aktiv → Entscheidung
        if pre_gate.active():
            action = pre_gate.handle_choice(q)
            if action == "go_order":
                reply = order_flow.start()
                print("🤖 ALESA:", reply, "\n")
                continue
            elif action == "go_qa":
                q = pre_gate.cached_user_query or q
                pre_gate.cached_user_query = ""
                # kein continue → an QA
            else:
                print("🤖 ALESA: Bitte antworte mit **bestellen** oder **beratung**.\n")
                continue

        # Einmalige Gate-Unterdrückung
        if pre_gate.suppress_next_gate:
            pre_gate.suppress_next_gate = False
        else:
            # C) Gate nur bei starker Kaufabsicht
            if pre_gate.should_prompt_gate(q):
                prompt = pre_gate.start(q)
                print("🤖 ALESA:", prompt, "\n")
                continue

        # D) Standard QA
        try:
            print("⏳ Verarbeite Anfrage …", flush=True)
            answer, cites = service.ask(q)
            print("✅ Fertig\n", flush=True)
        except ValueError as ve:
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
