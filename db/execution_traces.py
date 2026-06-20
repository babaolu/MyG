from __future__ import annotations

import base64
import gzip
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ExecutionTrace(BaseModel):
    trace_id: str
    session_id: str
    timestamp: str
    task_type: Literal["code_generation", "debug", "knowledge_query"]
    gpu_vendor: str
    gpu_model: str
    vulkan_version: str
    target_os: str
    user_request: str
    generated_code: str | None = None
    active_fix: str | None = None
    validation_passed: bool
    bug_classification: str | None = None
    bug_pattern_matched: str | None = None
    hypothesis_used: str | None = None
    hypothesis_probability: float | None = None
    outcome: Literal["success", "failure", "partial"]
    iterations_required: int
    time_to_resolve_seconds: float | None = None
    validation_stderr: str | None = None
    clang_tidy_warnings: int | None = None
    spirv_errors: int | None = None


class ExecutionTraceStore:
    def __init__(self, db_path: str):
        self.database_path = Path(db_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    gpu_vendor TEXT NOT NULL,
                    gpu_model TEXT NOT NULL,
                    vulkan_version TEXT NOT NULL,
                    target_os TEXT NOT NULL,
                    user_request TEXT NOT NULL,
                    generated_code TEXT,
                    active_fix TEXT,
                    validation_passed INTEGER NOT NULL,
                    bug_classification TEXT,
                    bug_pattern_matched TEXT,
                    hypothesis_used TEXT,
                    hypothesis_probability REAL,
                    outcome TEXT NOT NULL,
                    iterations_required INTEGER NOT NULL,
                    time_to_resolve_seconds REAL,
                    validation_stderr TEXT,
                    clang_tidy_warnings INTEGER,
                    spirv_errors INTEGER
                )
                """
            )

    def record(self, trace: ExecutionTrace) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO execution_traces(
                    trace_id, session_id, timestamp, task_type, gpu_vendor, gpu_model,
                    vulkan_version, target_os, user_request, generated_code, active_fix,
                    validation_passed, bug_classification, bug_pattern_matched,
                    hypothesis_used, hypothesis_probability, outcome, iterations_required,
                    time_to_resolve_seconds, validation_stderr, clang_tidy_warnings, spirv_errors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.trace_id,
                    trace.session_id,
                    trace.timestamp,
                    trace.task_type,
                    trace.gpu_vendor,
                    trace.gpu_model,
                    trace.vulkan_version,
                    trace.target_os,
                    trace.user_request,
                    _compress(trace.generated_code),
                    _compress(trace.active_fix),
                    1 if trace.validation_passed else 0,
                    trace.bug_classification,
                    trace.bug_pattern_matched,
                    trace.hypothesis_used,
                    trace.hypothesis_probability,
                    trace.outcome,
                    trace.iterations_required,
                    trace.time_to_resolve_seconds,
                    trace.validation_stderr,
                    trace.clang_tidy_warnings,
                    trace.spirv_errors,
                ),
            )

    def get_by_session(self, session_id: str) -> list[ExecutionTrace]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM execution_traces WHERE session_id = ? ORDER BY timestamp DESC",
                (session_id,),
            ).fetchall()
        return [_row_to_trace(row) for row in rows]

    def get_by_platform(
        self,
        gpu_vendor: str,
        vulkan_version: str,
        outcome: str | None = None,
        limit: int = 50,
    ) -> list[ExecutionTrace]:
        sql = """
            SELECT * FROM execution_traces
            WHERE gpu_vendor = ? AND vulkan_version = ?
        """
        params: list[object] = [gpu_vendor, vulkan_version]
        if outcome:
            sql += " AND outcome = ?"
            params.append(outcome)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_row_to_trace(row) for row in rows]

    def get_high_value_traces(
        self,
        min_iterations: int = 2,
        outcome: str = "success",
        limit: int = 20,
    ) -> list[ExecutionTrace]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM execution_traces
                WHERE iterations_required >= ? AND outcome = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (min_iterations, outcome, limit),
            ).fetchall()
        return [_row_to_trace(row) for row in rows]

    def get_pattern_hit_rate(self, symptom: str, gpu_vendor: str | None = None) -> float:
        if gpu_vendor:
            rows = self._fetch_rows(
                "SELECT outcome FROM execution_traces WHERE bug_pattern_matched = ? AND gpu_vendor = ?",
                (symptom, gpu_vendor),
            )
        else:
            rows = self._fetch_rows(
                "SELECT outcome FROM execution_traces WHERE bug_pattern_matched = ?",
                (symptom,),
            )
        if not rows:
            return 0.0
        successes = sum(1 for row in rows if row["outcome"] == "success")
        return successes / len(rows)

    def get_failures_since(self, since: datetime, limit: int = 200) -> list[ExecutionTrace]:
        since_text = since.astimezone(UTC).isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM execution_traces
                WHERE outcome != 'success' AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (since_text, limit),
            ).fetchall()
        return [_row_to_trace(row) for row in rows]

    def get_all(self, limit: int = 500) -> list[ExecutionTrace]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM execution_traces ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_trace(row) for row in rows]

    def total_traces(self) -> int:
        row = self._fetch_one("SELECT COUNT(*) AS count FROM execution_traces")
        return int(row["count"]) if row else 0

    def success_rate_by_platform(self) -> dict[str, float]:
        rows = self._fetch_all(
            """
            SELECT gpu_vendor,
                   SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) AS successes,
                   COUNT(*) AS total
            FROM execution_traces
            GROUP BY gpu_vendor
            """
        )
        return {row["gpu_vendor"]: row["successes"] / row["total"] for row in rows if row["total"]}

    def top_symptoms_resolved(self, limit: int = 10) -> list[str]:
        rows = self._fetch_all(
            """
            SELECT bug_pattern_matched, COUNT(*) AS count
            FROM execution_traces
            WHERE outcome = 'success' AND bug_pattern_matched IS NOT NULL
            GROUP BY bug_pattern_matched
            ORDER BY count DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [str(row["bug_pattern_matched"]) for row in rows]

    def average_iterations_to_resolve(self) -> float:
        row = self._fetch_one(
            """
            SELECT AVG(iterations_required) AS avg_iterations
            FROM execution_traces
            WHERE outcome = 'success'
            """
        )
        value = row["avg_iterations"] if row else None
        return float(value) if value is not None else 0.0

    def _fetch_one(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(sql, params).fetchone()

    def _fetch_all(self, sql: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(sql, params).fetchall()

    def _fetch_rows(self, sql: str, params: tuple[object, ...]) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(sql, params).fetchall()


def _compress(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.encode("utf-8")
    return base64.b64encode(gzip.compress(raw)).decode("ascii")


def _decompress(value: str | None) -> str | None:
    if not value:
        return None
    return gzip.decompress(base64.b64decode(value)).decode("utf-8")


def _row_to_trace(row: sqlite3.Row) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=row["trace_id"],
        session_id=row["session_id"],
        timestamp=row["timestamp"],
        task_type=row["task_type"],
        gpu_vendor=row["gpu_vendor"],
        gpu_model=row["gpu_model"],
        vulkan_version=row["vulkan_version"],
        target_os=row["target_os"],
        user_request=row["user_request"],
        generated_code=_decompress(row["generated_code"]),
        active_fix=_decompress(row["active_fix"]),
        validation_passed=bool(row["validation_passed"]),
        bug_classification=row["bug_classification"],
        bug_pattern_matched=row["bug_pattern_matched"],
        hypothesis_used=row["hypothesis_used"],
        hypothesis_probability=row["hypothesis_probability"],
        outcome=row["outcome"],
        iterations_required=int(row["iterations_required"]),
        time_to_resolve_seconds=row["time_to_resolve_seconds"],
        validation_stderr=row["validation_stderr"],
        clang_tidy_warnings=row["clang_tidy_warnings"],
        spirv_errors=row["spirv_errors"],
    )
