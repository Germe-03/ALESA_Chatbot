from __future__ import annotations

"""
Analytics/ML-Helfer fuer den Adminbereich.
Enthaelt Laden der Daten, Query-Clustering, Dokument- und Produkt-Trends.
LLM/RAG-Logik wird hier nicht beruehrt.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# ------------------------------------------------------------
# Daten-Layer
# ------------------------------------------------------------

def load_logs(path: str) -> pd.DataFrame:
    """Logs laden, Timestamp parsen, leere Texte entfernen."""
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "query_text" in df.columns:
        df["query_text"] = df["query_text"].fillna("").astype(str).str.strip()
        df = df[df["query_text"] != ""]
    return df


def load_documents(path: str) -> pd.DataFrame:
    """Dokumenten-Stammdaten laden."""
    return pd.read_csv(path)


def load_products(path: str) -> pd.DataFrame:
    """Produkt-Stammdaten laden."""
    return pd.read_csv(path)


# ------------------------------------------------------------
# Query-Cluster (Themenanalyse)
# ------------------------------------------------------------

SHORT_ANSWERS = {"ja", "nein", "ok", "okay", "danke", "fertig"}


def preprocess_for_clustering(text: str) -> bool:
    """Filtert Texte, die fuer Clustering nicht sinnvoll sind."""
    if not text:
        return False
    t = text.strip()
    if len(t) < 5:
        return False
    words = t.split()
    if len(words) < 2:
        return False
    low = t.lower()
    if low in SHORT_ANSWERS:
        return False
    if "@" in t:
        return False
    if all(ch.isdigit() or ch.isspace() or ch in ".,-/" for ch in t):
        return False
    return True


def _top_terms_per_cluster(kmeans: KMeans, vectorizer: TfidfVectorizer, top_k: int = 10):
    centers = kmeans.cluster_centers_
    feats = vectorizer.get_feature_names_out()
    res = {}
    for idx, row in enumerate(centers):
        top_idx = row.argsort()[::-1][:top_k]
        res[idx] = [feats[i] for i in top_idx]
    return res


def cluster_queries(logs_df: pd.DataFrame, max_queries: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clustert die letzten max_queries Nutzeranfragen."""
    if "timestamp" in logs_df.columns:
        logs_df = logs_df.sort_values("timestamp")
    recent = logs_df.tail(max_queries).copy()
    mask_valid = recent["query_text"].apply(preprocess_for_clustering)
    texts = recent.loc[mask_valid, "query_text"].tolist()

    logs_with_cluster = recent.copy()
    logs_with_cluster["cluster_id"] = -1

    if len(texts) < 2:
        clusters_overview = pd.DataFrame(columns=["cluster_id", "count", "top_terms", "example_queries"])
        return logs_with_cluster, clusters_overview

    stop_words = ["ist", "der", "die", "das", "ein", "eine", "und", "ich", "du", "wir", "ihr", "auch", "bitte"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, stop_words=stop_words)
    X = vectorizer.fit_transform(texts)

    k = min(8, max(2, int(len(texts) ** 0.5)))
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)

    # cluster_id zurueckschreiben
    valid_indices = logs_with_cluster.index[mask_valid].tolist()
    for idx, lbl in zip(valid_indices, labels):
        logs_with_cluster.at[idx, "cluster_id"] = int(lbl)

    # Cluster-Overview
    top_terms = _top_terms_per_cluster(kmeans, vectorizer, top_k=10)
    rows = []
    for cid in sorted(set(labels)):
        ids = [i for i, l in zip(valid_indices, labels) if l == cid]
        examples = logs_with_cluster.loc[ids, "query_text"].tolist()[:3]
        rows.append(
            {
                "cluster_id": int(cid),
                "count": int(len(ids)),
                "top_terms": top_terms.get(cid, []),
                "example_queries": examples,
            }
        )
    clusters_overview = pd.DataFrame(rows)
    return logs_with_cluster, clusters_overview


def print_cluster_overview(clusters_overview: pd.DataFrame) -> None:
    """Kleine Konsolenhilfe zum schnellen Check."""
    for _, row in clusters_overview.iterrows():
        print(f"Cluster {row['cluster_id']} ({row['count']}): {', '.join(row['top_terms'][:5])}")
        for ex in row["example_queries"]:
            print(f"  - {ex}")


# ------------------------------------------------------------
# Dokument-Trends
# ------------------------------------------------------------

def _trend_from_series(counts: pd.Series) -> float:
    """Einfache Steigung via linearer Regression ueber Index."""
    if len(counts) < 2:
        return 0.0
    x = np.arange(len(counts))
    y = counts.to_numpy()
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _label_trend(slope: float, threshold: float = 0.05) -> str:
    if slope > threshold:
        return "steigend"
    if slope < -threshold:
        return "fallend"
    return "stabil"


def compute_document_trends(logs_df: pd.DataFrame, documents_df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Aggregiert Dokument-Nutzung und schaetzt Trend."""
    df = logs_df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["document_id"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["document_id", "title", "category", "total_uses", "uses_last_30_days", "trend_slope", "trend_label"])

    df["period"] = df["timestamp"].dt.to_period(freq).dt.to_timestamp()
    agg = df.groupby(["document_id", "period"]).size().rename("uses").reset_index()

    rows = []
    cutoff = pd.Timestamp.now() - timedelta(days=30)
    for doc_id, grp in agg.groupby("document_id"):
        grp = grp.sort_values("period")
        slope = _trend_from_series(grp["uses"])
        uses_total = int(grp["uses"].sum())
        uses_30 = int(grp.loc[grp["period"] >= cutoff, "uses"].sum())
        rows.append(
            {
                "document_id": doc_id,
                "total_uses": uses_total,
                "uses_last_30_days": uses_30,
                "trend_slope": slope,
                "trend_label": _label_trend(slope),
            }
        )
    trends = pd.DataFrame(rows)
    trends = trends.merge(documents_df, on="document_id", how="left")
    return trends


def top_trending_documents(doc_trends_df: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    rising = doc_trends_df.sort_values("trend_slope", ascending=False).head(n)
    falling = doc_trends_df.sort_values("trend_slope", ascending=True).head(n)
    return rising, falling


# ------------------------------------------------------------
# Produkt-Trends & einfache Prognose
# ------------------------------------------------------------

def compute_product_trends(logs_df: pd.DataFrame, products_df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Aggregiert Produkt-Nachfrage (Bestellungen) und schaetzt Trend + Forecast."""
    df = logs_df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "intent" in df.columns:
        df = df[df["intent"] == "bestellung"]
    df = df[df["product_id"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["product_id", "name", "category", "total_orders", "orders_last_30_days", "trend_slope", "trend_label", "forecast_next_period"])

    df["period"] = df["timestamp"].dt.to_period(freq).dt.to_timestamp()
    agg = df.groupby(["product_id", "period"]).size().rename("orders").reset_index()

    rows = []
    cutoff = pd.Timestamp.now() - timedelta(days=30)
    for pid, grp in agg.groupby("product_id"):
        grp = grp.sort_values("period")
        slope = _trend_from_series(grp["orders"])
        total_orders = int(grp["orders"].sum())
        orders_30 = int(grp.loc[grp["period"] >= cutoff, "orders"].sum())
        # Forecast: letzte Periode oder Steigung addieren
        last_val = int(grp["orders"].iloc[-1])
        forecast = max(0, int(round(last_val + slope)))
        rows.append(
            {
                "product_id": pid,
                "total_orders": total_orders,
                "orders_last_30_days": orders_30,
                "trend_slope": slope,
                "trend_label": _label_trend(slope),
                "forecast_next_period": forecast,
            }
        )
    trends = pd.DataFrame(rows)
    trends = trends.merge(products_df, on="product_id", how="left")
    return trends


# ------------------------------------------------------------
# Admin-Overview
# ------------------------------------------------------------

def build_admin_ml_overview(logs_path: str, docs_path: str, products_path: str) -> dict:
    """Lädt Daten, berechnet Cluster und Trends, liefert JSON-freundliche Struktur."""
    logs_df = load_logs(logs_path)
    docs_df = load_documents(docs_path)
    prods_df = load_products(products_path)

    logs_with_cluster, clusters_overview = cluster_queries(logs_df)
    doc_trends = compute_document_trends(logs_df, docs_df)
    prod_trends = compute_product_trends(logs_df, prods_df)

    return {
        "clusters": {
            "overview": clusters_overview.to_dict(orient="records"),
            "logs": logs_with_cluster.to_dict(orient="records"),
        },
        "documents": {
            "trends": doc_trends.to_dict(orient="records"),
        },
        "products": {
            "trends": prod_trends.to_dict(orient="records"),
        },
    }


# ------------------------------------------------------------
# Main-Block zum schnellen Test
# ------------------------------------------------------------
if __name__ == "__main__":
    logs_path = "logs.csv"
    docs_path = "documents.csv"
    products_path = "products.csv"

    overview = build_admin_ml_overview(logs_path, docs_path, products_path)

    clusters = overview.get("clusters", {}).get("overview", [])
    print(f"Cluster: {len(clusters)}")
    if clusters:
        print("Beispiel-Cluster:", clusters[0])

    doc_trends = pd.DataFrame(overview.get("documents", {}).get("trends", []))
    if not doc_trends.empty:
        print("Top 5 steigende Dokumente:")
        print(doc_trends.sort_values("trend_slope", ascending=False).head(5)[["document_id", "title", "trend_label", "trend_slope"]])
        print("Top 5 fallende Dokumente:")
        print(doc_trends.sort_values("trend_slope", ascending=True).head(5)[["document_id", "title", "trend_label", "trend_slope"]])

    prod_trends = pd.DataFrame(overview.get("products", {}).get("trends", []))
    if not prod_trends.empty:
        print("Top 5 Produkte (steigend):")
        print(prod_trends.sort_values("trend_slope", ascending=False).head(5)[["product_id", "name", "trend_label", "trend_slope", "forecast_next_period"]])
        print("Top 5 Produkte (fallend):")
        print(prod_trends.sort_values("trend_slope", ascending=True).head(5)[["product_id", "name", "trend_label", "trend_slope", "forecast_next_period"]])
