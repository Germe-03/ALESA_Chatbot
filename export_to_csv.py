"""
Kleines Export-Skript fuer den ALESA-Chatbot:
- Liest Chat-Logs aus SQLite (data/logs/chat.db)
- Exportiert sie als data/logs.csv mit Spalten, die das Analytics-Modul erwartet.
- Erzeugt bei Bedarf Platzhalter fuer documents.csv und products.csv.

Anpassungshinweise:
- Falls der Tabellenname/Spalten in der DB anders sind, siehe Kommentare in `export_logs_db_to_csv`.
"""

import sqlite3
from pathlib import Path
import pandas as pd


def export_logs_db_to_csv(
    db_path: str = "data/logs/chat.db",
    csv_path: str = "data/logs.csv",
    table_name: str | None = None,
) -> None:
    """
    Exportiert Chat-Logs aus SQLite nach CSV.
    Erwartete Zielspalten: id, timestamp, query_text, intent, document_id, product_id, conversation_id.

    Falls Tabellennamen/Spalten abweichen:
    - Passe `source_table` an (Default: erste Tabelle 'interactions' falls vorhanden).
    - Passe das SELECT-Statement an, um Spalten zu mappen (z. B. ts -> timestamp, message -> query_text).
    """
    db_path = str(db_path)
    csv_path = str(csv_path)
    if not Path(db_path).exists():
        raise FileNotFoundError(f"DB nicht gefunden: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        # Tabellenname bestimmen (Fallback: interactions)
        if table_name is None:
            try:
                tables = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
                )
                # Suche nach einer plausiblen Tabelle
                candidates = [t for t in tables["name"].tolist() if t in {"interactions", "logs", "messages"}]
                source_table = candidates[0] if candidates else tables["name"].iloc[0]
            except Exception:
                source_table = "interactions"
        else:
            source_table = table_name

        # Hier Spalten der DB auf Zielspalten mappen.
        # Falls Spaltennamen abweichen, unten anpassen (z. B. ts -> timestamp, message -> query_text).
        query = f"""
        SELECT
            id,
            ts AS timestamp,
            message AS query_text,
            NULL AS intent,
            NULL AS document_id,
            NULL AS product_id,
            session_id AS conversation_id
        FROM {source_table}
        """
        df = pd.read_sql_query(query, conn)

        # Zielspalten sicherstellen
        target_cols = [
            "id",
            "timestamp",
            "query_text",
            "intent",
            "document_id",
            "product_id",
            "conversation_id",
        ]
        for col in target_cols:
            if col not in df.columns:
                df[col] = None
        df = df[target_cols]

        df.to_csv(csv_path, index=False)
        print(f"Export fertig: {csv_path} ({len(df)} Zeilen) aus Tabelle {source_table}")
    finally:
        conn.close()


def ensure_placeholder_csvs(
    docs_path: str = "data/documents.csv",
    products_path: str = "data/products.csv",
) -> None:
    """
    Erzeugt Platzhalter-CSV fuer Dokumente/Produkte, falls nicht vorhanden.
    """
    docs_file = Path(docs_path)
    if not docs_file.exists():
        docs_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"document_id": "DOC001", "title": "Beispiel-Dokument", "category": "Beispiel"},
            ]
        ).to_csv(docs_file, index=False)
        print(f"Placeholder erzeugt: {docs_file}")

    prods_file = Path(products_path)
    if not prods_file.exists():
        prods_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"product_id": "PRD001", "name": "Beispiel-Produkt", "category": "Beispiel"},
            ]
        ).to_csv(prods_file, index=False)
        print(f"Placeholder erzeugt: {prods_file}")


if __name__ == "__main__":
    export_logs_db_to_csv()
    ensure_placeholder_csvs()
