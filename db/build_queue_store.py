from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid the runtime circular import: db.build_queue_store ↔ agents.self_update
    from agents.self_update.monitors.khronos_monitor import UpdateDiff

from orchestrator.state import BuildCandidate


class BuildQueueStore:
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
                CREATE TABLE IF NOT EXISTS build_queue (
                    id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    cmake_preset TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS update_diffs (
                    update_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    current_version TEXT NOT NULL,
                    latest_version TEXT NOT NULL,
                    changelog TEXT NOT NULL,
                    new_extensions_json TEXT NOT NULL,
                    deprecated_patterns_json TEXT NOT NULL,
                    require_human_confirmation INTEGER NOT NULL,
                    confirmed INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )

    def add(self, candidate: BuildCandidate) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO build_queue(id, source_file, cmake_preset, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (candidate.id, candidate.source_file, candidate.cmake_preset, candidate.status, candidate.created_at),
            )

    def update_status(self, candidate_id: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE build_queue SET status = ? WHERE id = ?", (status, candidate_id))

    def list_queue(self) -> list[BuildCandidate]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM build_queue ORDER BY created_at DESC").fetchall()
        return [BuildCandidate.model_validate(dict(row)) for row in rows]

    def save_update_diff(self, diff: UpdateDiff) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO update_diffs(
                    update_id, source, current_version, latest_version, changelog,
                    new_extensions_json, deprecated_patterns_json, require_human_confirmation,
                    confirmed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    diff.update_id,
                    diff.source,
                    diff.current_version,
                    diff.latest_version,
                    diff.changelog,
                    str(diff.new_extensions),
                    str(diff.deprecated_patterns),
                    1 if diff.require_human_confirmation else 0,
                    None,
                    diff.created_at,
                ),
            )

    def pending_update_diffs(self) -> list[UpdateDiff]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM update_diffs WHERE confirmed IS NULL").fetchall()
        return [self._row_to_diff(row) for row in rows]

    def confirm_update_diff(self, update_id: str, confirmed: bool) -> UpdateDiff:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM update_diffs WHERE update_id = ?", (update_id,)).fetchone()
            if row is None:
                raise KeyError(update_id)
            connection.execute("UPDATE update_diffs SET confirmed = ? WHERE update_id = ?", (1 if confirmed else 0, update_id))
        diff = self._row_to_diff(row)
        diff.confirmed = confirmed
        return diff

    def _row_to_diff(self, row: sqlite3.Row) -> UpdateDiff:
        from agents.self_update.monitors.khronos_monitor import (
            UpdateDiff as _UpdateDiff,  # noqa: F811,F401
        )

        return _UpdateDiff(
            update_id=row["update_id"],
            source=row["source"],
            current_version=row["current_version"],
            latest_version=row["latest_version"],
            changelog=row["changelog"],
            new_extensions=_loads_list(row["new_extensions_json"]),
            deprecated_patterns=_loads_list(row["deprecated_patterns_json"]),
            require_human_confirmation=bool(row["require_human_confirmation"]),
            confirmed=bool(row["confirmed"]) if row["confirmed"] is not None else None,
            created_at=row["created_at"],
        )


def _loads_list(value: str) -> list[str]:
    try:
        import ast

        loaded = ast.literal_eval(value)
        return list(loaded) if isinstance(loaded, list) else []
    except (SyntaxError, ValueError):
        return []
