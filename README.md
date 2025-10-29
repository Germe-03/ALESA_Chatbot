# ALESA Chatbot (RAG)

ALESA ist ein produktionsnaher RAG‑Chatbot auf Basis von Google Vertex AI (Gemini 2.x) mit:
- Browser‑UI (blau/weiß), Session‑Chat `/chat`
- Vektor­datenbank (Chroma, persistent unter `data/vectorstore/`)
- Strukturierter Produkttabellen‑Suche (Artikelnummern, Maße)
- Bestell‑Assistent (OrderFlow) inkl. Mitarbeiterportal
- Reklamations‑Workflow (RMA) inkl. Statusabfrage und SLA‑Hinweisen

---

## Voraussetzungen

- Python 3.10+
- Google Cloud Projekt mit aktivierter Vertex AI API
- Service‑Account‑Key (JSON) mit Rollen:
  - Vertex AI User
  - Service Usage Consumer
  - optional: Storage Object Viewer

### .env Variablen

```
GCP_PROJECT=dein-projekt
GCP_LOCATION=us-central1           # Generatives Modell (z. B. europe-west6)
GEMINI_MODEL=gemini-2.0-pro        # oder gemini-2.0-flash
GOOGLE_APPLICATION_CREDENTIALS=C:\\pfad\\zu\\creds.json
SYSTEM_PROMPT=Du bist ALESA, ein hilfreicher Assistent.
# optional (Embeddings)
EMBEDDING_LOCATION=us-central1
EMBEDDING_MODEL=text-embedding-004
```

---

## Installation

```
pip install -r requirements.txt
```

---

## Starten

Browser‑API starten (empfohlen):

```
uvicorn src.alesa_bot.web_api:app --host 127.0.0.1 --port 8080
```

Öffne anschließend `http://127.0.0.1:8080/`.

CLI‑Variante (optional):

```
python -m src.alesa_bot.app
```

Docker (optional):

```
docker build -t alesa .
docker run -p 8080:8080 --env-file .env -v C:\\pfad\\zu\\creds.json:/app/creds.json alesa
```

---

## Nutzung (Browser‑UI)

- Fragestellungen zu Produkten/Dokumenten stellen (RAG). Quellen werden angezeigt.
- Produktmaße zu Artikelnummern (z. B. `6042.0206`) werden als Tabelle ausgegeben.
- Nach einer Produktantwort fragt der Bot automatisch: „Moechten Sie dieses Produkt bestellen?“
  - „ja“ oder exakt „bestellen“ startet den Bestell‑Assistenten.
  - Exakt „bestellen“ startet jederzeit direkt den Bestell‑Flow. „bestellen“ in einem Satz löst nichts aus.

### Bestell‑Assistent (OrderFlow)
- Schritte: Kundendaten → Artikel (Artikelnummer, Menge; Befehle: `neuer artikel`, `fertig`) → Kommentar → Bestätigung.
- Persistenz: `data/orders/orders.jsonl`
- Mitarbeiterportal: `http://127.0.0.1:8080/admin` (Liste der Bestellungen)

### Reklamationen (RMA)
- Start: Exakt `reklamation` (oder `retoure`, `umtausch`, `garantie`).
- Schritte: E‑Mail → Referenzen → Artikel → Problem → Belege → bevorzugte Lösung → Bestätigung.
- Persistenz: `data/orders/rmas.jsonl`
- Statusabfrage im Chat: `status RMA-<id>` (z. B. `status RMA-20250101093000123`)

---

## Architektur‑Überblick

- Retrieval/Index:
  - In‑Memory‑Indexer für `.txt/.md/.pdf` (pypdf) → Seiten‑Text
  - HybridRetriever (BM25‑lite + Embeddings) und Chroma Vektorstore (persistent)
  - Artikelnummer‑Booster (exakte Matches für `dddd.dddd`/`dddddddd`)
- Produkttabellen:
  - CSV ingest aus `data/products/**/*.csv` oder `data/processed/products/**/*.csv`
  - Heuristische Zeilen‑Extraktion aus PDF‑Seiten als Fallback
- LLM/Prompts: Vertex AI, Antwort strikte Quellenbindung
- Flows: OrderFlow (Bestellung), RMAFlow (Reklamation)

---

## API‑Endpunkte (Auszug)

- `POST /chat` — Session‑Chat (Form: `session`, `message`)
- `GET /` — Browser‑UI
- `GET /health`
- `GET /admin` — Mitarbeiterportal (Bestellungen)
- `GET /admin/orders` — Liste Bestellungen (JSON)
- `POST /admin/order` — Bestellung anlegen (JSON)
- Legacy: `POST /ask` — einfache Q&A (ohne Session/Flows)

---

## Daten & Pfade

- `data/vectorstore/` — Chroma (persistent)
- `data/orders/orders.jsonl` — Bestellungen
- `data/orders/rmas.jsonl` — Reklamationen (RMA)
- `data/raw/` — Rohdokumente (PDF, TXT)
- `data/processed/` — aufbereitete Dateien (optional)

Der Vektorindex wird beim Start automatisch (re)gebaut. Lege neue Dateien in `data/` ab und starte neu.

---

## CSV für Produkttabellen (optional, empfohlen)

Lege unter `data/products/` oder `data/processed/products/` CSVs ab. Erwartete Spalten (Beispiele, Groß-/Kleinschreibung egal):

```
artikel,d1,b,b2,nuttiefe,d2,d3,zahnform,aufnahme
6042.0206, ...
```

Artikelnummern werden normalisiert (`6042.0206`, `6042-0206`, `60420206`).

---

## Fehlerbehebung

- 500 beim `/chat`: Prüfe `.env`, GCP‑Zugang und Server‑Logs; stelle sicher, dass Sessions initialisiert sind.
- Keine Produktmaße: CSV ergänzen oder sicherstellen, dass die Katalog‑PDFs Text enthalten (kein Scan) und `max_pdf_pages` ausreichend hoch ist.
- Umlaute im Terminal: UTF‑8‑Ausgabe/Font verwenden; im Browser‑UI kein Problem.
- Sicherheit: Standard‑CORS ist `*`. Für Produktivbetrieb einschränken und Auth vorsehen.

---

## Lizenz / Hinweise

Dieses Repository enthält Beispielcode für einen internen Chatbot‑Prototyp. Prüfe Rechte an Dokumenten/Daten vor Produktionseinsatz.
