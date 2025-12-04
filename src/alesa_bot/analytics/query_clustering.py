from __future__ import annotations

from typing import Any, Dict, List
from collections import defaultdict
import re

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.alesa_bot.settings import load_config
from src.alesa_bot.services.chat_logger import ChatLogger

# Maximalzahl der Anfragen, die pro Lauf geclustert werden (neueste zuerst)
MAX_QUERIES_FOR_CLUSTERING = 1000


def _get_logger() -> ChatLogger:
    cfg = load_config()
    db_path = cfg.paths.data_root / "logs" / "chat.db"
    return ChatLogger(db_path=db_path)


def load_query_texts(max_sessions: int = 5000, max_messages_per_session: int = 2000) -> List[Dict[str, Any]]:
    """
    Holt Nutzeranfragen aus dem bestehenden Chat-Log (nur Rolle 'user').
    Gibt eine Liste von Dicts mit id, text, timestamp zurueck.
    """
    logger = _get_logger()
    sessions = logger.list_sessions(limit=max_sessions)
    records: List[Dict[str, Any]] = []
    for sess in sessions:
        sid = sess.get("session_id")
        msgs = logger.list_messages(session_id=sid, limit=max_messages_per_session)
        for m in msgs:
            if (m.get("role") or "").lower() != "user":
                continue
            txt = (m.get("message") or "").strip()
            if not txt:
                continue
            records.append(
                {
                    "id": m.get("id") or f"{sid}:{m.get('ts','')}",
                    "text": txt,
                    "timestamp": m.get("ts"),
                }
            )
    # Neueste zuerst begrenzen
    records_sorted = sorted(records, key=lambda r: r.get("timestamp") or "", reverse=True)
    return records_sorted[:MAX_QUERIES_FOR_CLUSTERING]


def is_useful_for_clustering(text: str) -> bool:
    """
    Filtert triviale/kurze Inhalte (z. B. Zahlen, E-Mails, Ein-Wort-Antworten),
    um sauberere Cluster zu erhalten.
    """
    if not text:
        return False
    t = (text or "").strip()
    if len(t) < 5:
        return False
    words = t.split()
    if len(words) < 2:
        return False
    low = t.lower()
    short_answers = {"ja", "nein", "ok", "okay", "danke", "fertig", "passt", "klar"}
    if low in short_answers:
        return False
    if "@" in t:
        return False
    if re.fullmatch(r"[0-9\\.\\-\\/ ]+", t):
        return False
    return True


def _top_terms_per_cluster(kmeans: KMeans, vectorizer: TfidfVectorizer, top_k: int = 10) -> Dict[int, List[str]]:
    centers = kmeans.cluster_centers_
    feature_names = vectorizer.get_feature_names_out()
    result: Dict[int, List[str]] = {}
    for idx, row in enumerate(centers):
        top_idx = row.argsort()[::-1][:top_k]
        result[idx] = [feature_names[i] for i in top_idx]
    return result


def find_optimal_k(X, n_texts: int, k_min: int = 2, k_max: int = 10) -> int:
    """
    Waehlt k per Silhouettenwert; bremst k bei wenig Daten.
    """
    if n_texts <= 2:
        return 1
    k_upper = min(k_max, max(2, int(n_texts ** 0.5)))
    if k_upper < k_min:
        k_upper = k_min

    best_k = k_min
    best_score = -1.0
    for k in range(k_min, k_upper + 1):
        if k >= n_texts:
            break
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        # Stichproben-Silhouette bei grossen N, um Laufzeit zu sparen
        sample_size = min(1000, n_texts)
        try:
            score = silhouette_score(X, labels, sample_size=sample_size, random_state=42)
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_k = k
    return max(1, best_k)


def cluster_queries(query_records: List[Dict[str, Any]], n_clusters: int = 8) -> Dict[str, Any]:
    """
    Fuehrt TF-IDF + KMeans Clustering auf Nutzerfragen durch.
    """
    texts = []
    ids = []
    for rec in query_records:
        txt = (rec.get("text") or "").strip()
        if not txt or not is_useful_for_clustering(txt):
            continue
        texts.append(txt)
        ids.append(rec.get("id"))

    if len(texts) < 2:
        # zu wenig Daten fuer k>=2
        return {"assignments": [], "clusters": []}

    # TF-IDF mit kleinen Stopwoertern und min_df gegen Rauschen
    stop_words = ["ist", "der", "die", "das", "ein", "eine", "und", "ich", "du", "wir", "ihr", "auch", "bitte"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, stop_words=stop_words)
    X = vectorizer.fit_transform(texts)

    # Dynamische k-Wahl (Begrenzung auf wenige Cluster bei wenig Daten)
    k_cap = min(8, max(2, int(len(texts) ** 0.5)))
    optimal_k = find_optimal_k(X, n_texts=len(texts), k_min=2, k_max=max(2, k_cap))
    kmeans = KMeans(n_clusters=min(optimal_k, len(texts)), n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)

    # Mini-Cluster behandeln
    min_cluster_size = 3
    cluster_sizes = defaultdict(int)
    for c_id in labels:
        cluster_sizes[int(c_id)] += 1

    # Ziel-Cluster fuer kleine Gruppen bestimmen (naechster grosser Cluster)
    big_clusters = {cid for cid, cnt in cluster_sizes.items() if cnt >= min_cluster_size}
    if not big_clusters:
        big_clusters = set(cluster_sizes.keys())

    if big_clusters:
        distances = kmeans.transform(X)
        for i, c_id in enumerate(labels):
            if cluster_sizes[int(c_id)] < min_cluster_size:
                # naechster grosser Cluster (außer eigener)
                best = None
                best_dist = float("inf")
                for bc in big_clusters:
                    if bc == c_id:
                        continue
                    d = distances[i, bc]
                    if d < best_dist:
                        best_dist = d
                        best = bc
                labels[i] = best if best is not None else -1

    assignments = [{"id": ids[i], "cluster_id": int(labels[i])} for i in range(len(ids))]

    # Sammle Beispiele pro Cluster
    cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, c_id in enumerate(labels):
        cluster_to_indices[int(c_id)].append(idx)

    top_terms = _top_terms_per_cluster(kmeans, vectorizer, top_k=10)
    clusters = []
    all_cluster_ids = sorted(cluster_to_indices.keys())
    for cid in all_cluster_ids:
        if cid == -1:
            # Noise/Reste
            idxs = cluster_to_indices.get(cid, [])
            examples = [texts[i] for i in idxs[:3]]
            clusters.append(
                {
                    "cluster_id": -1,
                    "count": len(idxs),
                    "example_queries": examples,
                    "top_terms": [],
                    "label": "Noise/Rest",
                }
            )
            continue
        idxs = cluster_to_indices.get(cid, [])
        examples = [texts[i] for i in idxs[:3]]
        clusters.append(
            {
                "cluster_id": cid,
                "count": len(idxs),
                "example_queries": examples,
                "top_terms": top_terms.get(cid, []),
            }
        )

    return {
        "assignments": assignments,
        "clusters": clusters,
    }


def get_cluster_overview(n_clusters: int = 8) -> Dict[str, Any]:
    """
    High-Level Helper fuer Admin: liest Logs und liefert Cluster-Uebersicht.
    """
    records = load_query_texts()
    return cluster_queries(records, n_clusters=n_clusters)


# Optionaler Cache, um beim Start einmalig zu clustern und Ergebnisse bereitzuhalten
_LAST_OVERVIEW: Dict[str, Any] | None = None


def build_clusters_cache(n_clusters: int = 8) -> Dict[str, Any]:
    global _LAST_OVERVIEW
    _LAST_OVERVIEW = get_cluster_overview(n_clusters=n_clusters)
    return _LAST_OVERVIEW


def get_cached_clusters(n_clusters: int = 8) -> Dict[str, Any]:
    if _LAST_OVERVIEW is None:
        return build_clusters_cache(n_clusters=n_clusters)
    return _LAST_OVERVIEW


if __name__ == "__main__":
    overview = get_cluster_overview(n_clusters=8)
    print(f"Cluster erzeugt: {len(overview.get('clusters', []))}")
    for c in overview.get("clusters", [])[:3]:
        print(f"Cluster {c['cluster_id']} - {c['count']} Anfragen - Top Begriffe: {', '.join(c['top_terms'][:5])}")
