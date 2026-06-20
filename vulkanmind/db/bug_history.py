from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class BugHistoryStore:
    def __init__(self, database_path: str | Path = "data/vulkanmind.sqlite3") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS bug_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    hypotheses_json TEXT NOT NULL,
                    active_fix TEXT,
                    created_at REAL NOT NULL
                )
                """
            )

    def record(self, session_id: str, classification: str, hypotheses: list[dict], active_fix: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO bug_history(session_id, classification, hypotheses_json, active_fix, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, classification, str(hypotheses), active_fix, time.time()),
            )

    def history(self, session_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM bug_history WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]
