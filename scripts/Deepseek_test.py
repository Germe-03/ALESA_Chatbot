# scripts/Deepseek_test.py
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Projekt-Root = eine Ebene über "scripts"
ROOT = Path(__file__).resolve().parents[1]

# Bevorzugt ".env", fallback auf "API.env"
dotenv_path = (ROOT / ".env")
if not dotenv_path.exists():
    alt = ROOT / "API.env"
    if alt.exists():
        dotenv_path = alt

# .env laden (Fehler bewusst nicht verschlucken)
loaded = load_dotenv(dotenv_path=str(dotenv_path), override=True)

key = os.getenv("DEEPSEEK_API_KEY")
if not key:
    # Debug-Hinweise ausgeben
    raise RuntimeError(
        f"DEEPSEEK_API_KEY nicht gefunden.\n"
        f"Erwartete Datei: {dotenv_path}\n"
        f"Beispielinhalt (ohne Anführungszeichen):\n"
        f"DEEPSEEK_API_KEY=sk-...dein_key...\n"
        f"Tipp: In PyCharm → Run/Debug Configurations → Environment variables "
        f"DEEPSEEK_API_KEY=sk-... setzen, falls du keine .env nutzen willst."
    )

# Manche OpenAI-Version erwartet OPENAI_API_KEY – setzen wir zusätzlich:
os.environ["OPENAI_API_KEY"] = key

client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

resp = client.chat.completions.create(
    model="deepseek-chat",  # oder "deepseek-reasoner"
    messages=[
        {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
        {"role": "user", "content": "Erkläre mir Transformers in 3 Sätzen."}
    ],
    temperature=0.7,
)
print(resp.choices[0].message.content)
