from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any, Iterable, Optional


class ChatLogger:
    """Einfacher SQLite-Logger für Chat- und Order-Ereignisse."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT,
              ts TEXT,
              role TEXT,
              message TEXT,
              sources TEXT,
              keywords TEXT,
              meta TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              session_id TEXT PRIMARY KEY,
              created_at TEXT,
              last_seen TEXT,
              meta TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS orders_log (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              order_id TEXT,
              session_id TEXT,
              created_at TEXT,
              payload TEXT,
              items TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    # -------- Lese-APIs fuer Admin --------

    def list_sessions(self, limit: int = 200) -> list[dict[str, Any]]:
        """Gibt gespeicherte Sessions mit rudimentaerer Statistik zurueck."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT s.session_id, s.created_at, s.last_seen,
                       COALESCE((SELECT COUNT(1) FROM interactions i WHERE i.session_id = s.session_id), 0) AS messages
                FROM sessions s
                ORDER BY datetime(s.last_seen) DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
            conn.close()
            return [
                {
                    "session_id": r[0],
                    "created_at": r[1],
                    "last_seen": r[2],
                    "messages": r[3],
                }
                for r in rows
            ]
        except Exception:
            return []

    def list_messages(self, session_id: str, limit: int = 400) -> list[dict[str, Any]]:
        """Liest Chatverlauf einer Session (chronologisch)."""
        if not session_id:
            return []
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, ts, role, message, sources, keywords, meta
                FROM interactions
                WHERE session_id = ?
                ORDER BY datetime(ts) ASC
                LIMIT ?
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()
            conn.close()
            return [
                {
                    "id": r[0],
                    "ts": r[1],
                    "role": r[2],
                    "message": r[3],
                    "sources": self._safe_load(r[4]),
                    "keywords": self._safe_load(r[5]),
                    "meta": self._safe_load(r[6]),
                }
                for r in rows
            ]
        except Exception:
            return []
    def _now(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def _dump(self, val: Any) -> Optional[str]:
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return None

    def _safe_load(self, val: Any) -> Any:
        if val is None:
            return None
        try:
            return json.loads(val)
        except Exception:
            return None

    def log_interaction(
        self,
        session_id: str,
        role: str,
        message: str,
        sources: Optional[Iterable[Any]] = None,
        keywords: Optional[Iterable[str]] = None,
        meta: Optional[dict] = None,
    ) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            ts = self._now()
            cur.execute(
                """
                INSERT INTO interactions (session_id, ts, role, message, sources, keywords, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    ts,
                    role,
                    message,
                    self._dump(list(sources)) if sources is not None else None,
                    self._dump(list(keywords)) if keywords is not None else None,
                    self._dump(meta) if meta is not None else None,
                ),
            )
            cur.execute(
                """
                INSERT INTO sessions (session_id, created_at, last_seen)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET last_seen=excluded.last_seen
                """,
                (session_id, ts, ts),
            )
            conn.commit()
            conn.close()
        except Exception:
            # Logging darf nicht den Chat blockieren
            return

    def log_order(self, session_id: str, order_id: str, payload: dict, items: Optional[Iterable[Any]] = None) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            ts = self._now()
            cur.execute(
                """
                INSERT INTO orders_log (order_id, session_id, created_at, payload, items)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    session_id,
                    ts,
                    self._dump(payload),
                    self._dump(list(items)) if items is not None else None,
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            return
