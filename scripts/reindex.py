from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
import re

# Projektpfade (robust relativ zu /scripts)
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
VECTOR_DIR = ROOT / "data" / "vectorstore"

# RAG-Parameter (leicht größer für juristische Texte)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 180
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def infer_category(path_str: str) -> str:
    p = path_str.replace("\\", "/").lower()
    if "/legal/" in p or "/recht/" in p:
        return "legal"
    if "/products/" in p or "/produkt" in p:
        return "products"
    if "/services/" in p or "/service/" in p:
        return "services"
    if "/company/" in p or "/unternehmen/" in p:
        return "company"
    return "other"

def infer_language(path_str: str) -> str:
    p = path_str.replace("\\", "/").lower()
    # Ordner- und Dateinamens-Hinweise
    if "/de/" in p or " deutsch" in p or "_de" in p or "-de" in p:
        return "de"
    if "/en/" in p or " english" in p or "_en" in p or "-en" in p:
        return "en"
    if re.search(r"(?:^|[_\- ])de(?:[_\- ]|$)", p):
        return "de"
    if re.search(r"(?:^|[_\- ])en(?:[_\- ]|$)", p):
        return "en"
    return "de"  # Default

def load_docs():
    for p in RAW_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}:
            if p.suffix.lower() == ".pdf":
                text = ""
                try:
                    for page in PdfReader(str(p)).pages:
                        text += page.extract_text() or ""
                except Exception:
                    continue
            else:
                text = p.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                yield str(p), text

def main():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    emb = SentenceTransformer(EMB_MODEL)
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))

    # Für den Neuaufbau: alte Collection löschen (falls vorhanden)
    try:
        client.delete_collection("alesa-rag")
    except Exception:
        pass

    coll = client.get_or_create_collection(
        name="alesa-rag",
        metadata={"hnsw:space": "cosine"},
    )

    ids, docs, metas = [], [], []

    for path, text in load_docs():
        category = infer_category(path)
        language = infer_language(path)
        chunks = splitter.split_text(text)
        for i, ch in enumerate(chunks):
            ids.append(f"{path}#{i}")
            docs.append(ch)
            metas.append({
                "source": path,
                "category": category,
                "language": language,
            })

    if not docs:
        print("Keine Dokumente gefunden.")
        return

    print(f"Erzeuge Embeddings für {len(docs)} Chunks…")
    embs = emb.encode(docs, normalize_embeddings=True).tolist()
    coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    print(f"Fertig. Chunks im Index: {coll.count()}")

if __name__ == "__main__":
    main()
