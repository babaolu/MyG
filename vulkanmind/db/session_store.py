from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from uuid import uuid4


class SessionStore:
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
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_request TEXT NOT NULL,
                    target_platform_json TEXT,
                    created_at REAL NOT NULL
                )
                """
            )

    def create_session(self, user_request: str, target_platform_declared: dict | None = None) -> str:
        session_id = str(uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions(session_id, user_request, target_platform_json, created_at) VALUES (?, ?, ?, ?)",
                (session_id, user_request, json.dumps(target_platform_declared or {}), time.time()),
            )
        return session_id

    def get_session(self, session_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return dict(row) if row else None
