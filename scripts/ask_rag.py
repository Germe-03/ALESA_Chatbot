from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, sys
import chromadb
from sentence_transformers import SentenceTransformer

# Projektpfade
ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "data" / "vectorstore"

# Modelle & Retrieval
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"           # CPU-geeignet (für GPU: Mistral 7B)
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 8                                       # für juristische Antworten etwas höher

# Einfache Erkennung „rechtliche Frage?“
LEGAL_KEYWORDS = [
    "agb","lieferbeding","widerruf","rücktritt","garantie","haftung",
    "datenschutz","zahlung","rabatt","mindestbestellwert","mbw","retoure",
    "gewährleistung","widerspruch","vertrag","bedingungen"
]

def is_legal_question(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in LEGAL_KEYWORDS)

def retrieve(question: str):
    emb = SentenceTransformer(EMB_MODEL)
    q = emb.encode([question], normalize_embeddings=True).tolist()[0]

    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    coll = client.get_or_create_collection("alesa-rag")

    where = None
    if is_legal_question(question):
        # Chroma-Filter mit Operatoren (neue API)
        where = {
            "$and": [
                {"category": {"$eq": "legal"}},
                {"language": {"$eq": "de"}},
            ]
        }

    res = coll.query(query_embeddings=[q], n_results=TOP_K, where=where)
    ctx = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        ctx.append((doc, meta.get("source", "")))
    return ctx

def main():
    question = " ".join(sys.argv[1:]) or "Wie hoch ist der Mindestbestellwert (DE/CH/Export)?"
    ctx = retrieve(question)

    context_block = "\n\n".join(
        f"[{i+1}] {t}\nQuelle: {s}"
        for i, (t, s) in enumerate(ctx, 1)
    ) or "(kein Kontext)"

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        dtype="auto"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    messages = [
        {"role": "system", "content": (
            "Du bist ein sachlicher Assistent für ALESA AG. "
            "Antworte ausschließlich mit Informationen aus dem Kontext. "
            "Wenn die Information nicht im Kontext steht, antworte exakt: 'Nicht im Kontext gefunden.' "
            "Gib Beträge und Währungen exakt wieder und nenne am Ende die Quellenpfade."
        )},
        {"role": "user", "content": f"Kontext:\n{context_block}\n\nFrage: {question}"}
    ]

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    in_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,           # deterministisch
            temperature=0.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

    new_tokens = out[0][in_len:]
    answer = tok.decode(new_tokens, skip_special_tokens=True).strip()

    print("\n--- Antwort ---\n" + answer)
    print("\n--- Quellen ---")
    for _, src in ctx:
        print("-", src)

if __name__ == "__main__":
    main()
