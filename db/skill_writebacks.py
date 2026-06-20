from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class VulkanSkill(BaseModel):
    skill_id: str
    created_at: str
    last_used_at: str | None = None
    last_improved_at: str | None = None
    name: str
    symptom: str
    domain: str
    gpu_vendor: str | None = None
    vulkan_version_min: str
    target_os: str | None = None
    fix_procedure: str
    code_template: str | None = None
    cmake_snippet: str | None = None
    validation_step: str
    times_applied: int = 0
    times_successful: int = 0
    confidence: Literal["low", "medium", "high", "trusted"]
    status: Literal["active", "under_review", "retired"] = "active"
    retirement_reason: str | None = None


class SkillStore:
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
                CREATE TABLE IF NOT EXISTS vulkan_skills (
                    skill_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    last_improved_at TEXT,
                    name TEXT NOT NULL,
                    symptom TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    gpu_vendor TEXT,
                    vulkan_version_min TEXT NOT NULL,
                    target_os TEXT,
                    fix_procedure TEXT NOT NULL,
                    code_template TEXT,
                    cmake_snippet TEXT,
                    validation_step TEXT NOT NULL,
                    times_applied INTEGER NOT NULL,
                    times_successful INTEGER NOT NULL,
                    confidence TEXT NOT NULL,
                    status TEXT NOT NULL,
                    retirement_reason TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS curation_reports (
                    report_id TEXT PRIMARY KEY,
                    run_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS human_review_queue (
                    item_id TEXT PRIMARY KEY,
                    item_type TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    resolved INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_refinement_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    agent_target TEXT NOT NULL,
                    current_prompt_excerpt TEXT NOT NULL,
                    proposed_change TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_runtime_overrides (
                    agent_target TEXT PRIMARY KEY,
                    prompt_override TEXT NOT NULL,
                    proposal_id TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pattern_writebacks (
                    symptom TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    skill_id TEXT NOT NULL,
                    gpu_vendor TEXT,
                    vulkan_version TEXT NOT NULL,
                    hit_rate REAL NOT NULL DEFAULT 0,
                    under_review INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def save(self, skill: VulkanSkill) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO vulkan_skills(
                    skill_id, created_at, last_used_at, last_improved_at, name, symptom,
                    domain, gpu_vendor, vulkan_version_min, target_os, fix_procedure,
                    code_template, cmake_snippet, validation_step, times_applied,
                    times_successful, confidence, status, retirement_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    skill.skill_id,
                    skill.created_at,
                    skill.last_used_at,
                    skill.last_improved_at,
                    skill.name,
                    skill.symptom,
                    skill.domain,
                    skill.gpu_vendor,
                    skill.vulkan_version_min,
                    skill.target_os,
                    skill.fix_procedure,
                    skill.code_template,
                    skill.cmake_snippet,
                    skill.validation_step,
                    skill.times_applied,
                    skill.times_successful,
                    skill.confidence,
                    skill.status,
                    skill.retirement_reason,
                ),
            )

    def get_by_symptom(
        self,
        symptom: str,
        gpu_vendor: str | None = None,
        vulkan_version: str | None = None,
    ) -> list[VulkanSkill]:
        rows = self._fetch_all(
            "SELECT * FROM vulkan_skills WHERE symptom = ? AND status = 'active'",
            (symptom,),
        )
        skills = [_row_to_skill(row) for row in rows]
        if gpu_vendor:
            skills = [skill for skill in skills if skill.gpu_vendor in {None, gpu_vendor}]
        if vulkan_version:
            skills = [skill for skill in skills if _version_gte(vulkan_version, skill.vulkan_version_min)]
        return sorted(skills, key=lambda item: 0 if item.gpu_vendor == gpu_vendor else 1)

    def get_trusted_skills(self, domain: str | None = None) -> list[VulkanSkill]:
        if domain:
            rows = self._fetch_all(
                "SELECT * FROM vulkan_skills WHERE confidence = 'trusted' AND domain = ? AND status = 'active'",
                (domain,),
            )
        else:
            rows = self._fetch_all("SELECT * FROM vulkan_skills WHERE confidence = 'trusted' AND status = 'active'")
        return [_row_to_skill(row) for row in rows]

    def increment_usage(self, skill_id: str, successful: bool) -> None:
        row = self._fetch_one("SELECT * FROM vulkan_skills WHERE skill_id = ?", (skill_id,))
        if row is None:
            return
        skill = _row_to_skill(row)
        skill.times_applied += 1
        if successful:
            skill.times_successful += 1
        skill.last_used_at = _now()
        if skill.times_successful >= 3:
            skill.confidence = "trusted"
        self.save(skill)

    def get_stale_skills(self, days_unused: int = 90) -> list[VulkanSkill]:
        cutoff = datetime.now(UTC) - timedelta(days=days_unused)
        stale: list[VulkanSkill] = []
        for skill in self.get_all_active():
            if skill.last_used_at is None:
                stale.append(skill)
                continue
            try:
                last_used = datetime.fromisoformat(skill.last_used_at)
            except ValueError:
                continue
            if last_used.astimezone(UTC) < cutoff:
                stale.append(skill)
        return stale

    def retire(self, skill_id: str, reason: str) -> None:
        skill = self.get_by_id(skill_id)
        if skill is None:
            return
        skill.status = "retired"
        skill.retirement_reason = reason
        self.save(skill)

    def get_all_active(
        self,
        gpu_vendor: str | None = None,
        domain: str | None = None,
        confidence: str | None = None,
    ) -> list[VulkanSkill]:
        sql = "SELECT * FROM vulkan_skills WHERE status = 'active'"
        params: list[object] = []
        if gpu_vendor:
            sql += " AND (gpu_vendor IS NULL OR gpu_vendor = ?)"
            params.append(gpu_vendor)
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        if confidence:
            sql += " AND confidence = ?"
            params.append(confidence)
        rows = self._fetch_all(sql, tuple(params))
        return [_row_to_skill(row) for row in rows]

    def get_by_id(self, skill_id: str) -> VulkanSkill | None:
        row = self._fetch_one("SELECT * FROM vulkan_skills WHERE skill_id = ?", (skill_id,))
        return _row_to_skill(row) if row else None

    def save_curation_report(self, report: BaseModel) -> None:
        payload = json.dumps(report.model_dump())
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO curation_reports(report_id, run_at, payload_json) VALUES (?, ?, ?)",
                (report.run_at, report.run_at, payload),
            )

    def save_human_review(self, item_id: str, item_type: str, reason: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO human_review_queue(item_id, item_type, reason, created_at, resolved)
                VALUES (?, ?, ?, ?, 0)
                """,
                (item_id, item_type, reason, _now()),
            )

    def get_human_review(self) -> list[dict[str, str]]:
        rows = self._fetch_all("SELECT * FROM human_review_queue WHERE resolved = 0 ORDER BY created_at DESC")
        return [dict(row) for row in rows]

    def resolve_human_review(self, item_id: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE human_review_queue SET resolved = 1 WHERE item_id = ?", (item_id,))

    def save_prompt_proposal(self, proposal: BaseModel) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO prompt_refinement_proposals(
                    proposal_id, created_at, agent_target, current_prompt_excerpt,
                    proposed_change, rationale, evidence_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal.proposal_id,
                    proposal.created_at,
                    proposal.agent_target,
                    proposal.current_prompt_excerpt,
                    proposal.proposed_change,
                    proposal.rationale,
                    json.dumps(proposal.evidence),
                    proposal.status,
                ),
            )

    def get_prompt_proposals(self, status: str | None = None) -> list[BaseModel]:
        if status:
            rows = self._fetch_all(
                "SELECT * FROM prompt_refinement_proposals WHERE status = ? ORDER BY created_at DESC",
                (status,),
            )
        else:
            rows = self._fetch_all("SELECT * FROM prompt_refinement_proposals ORDER BY created_at DESC")
        return [_row_to_prompt_proposal(row) for row in rows]

    def approve_prompt_proposal(self, proposal_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE prompt_refinement_proposals SET status = 'approved' WHERE proposal_id = ?",
                (proposal_id,),
            )

    def reject_prompt_proposal(self, proposal_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE prompt_refinement_proposals SET status = 'rejected' WHERE proposal_id = ?",
                (proposal_id,),
            )

    def apply_prompt_proposal(self, proposal: BaseModel) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO prompt_runtime_overrides(
                    agent_target, prompt_override, proposal_id, applied_at
                ) VALUES (?, ?, ?, ?)
                """,
                (proposal.agent_target, proposal.proposed_change, proposal.proposal_id, _now()),
            )
            connection.execute(
                "UPDATE prompt_refinement_proposals SET status = 'applied' WHERE proposal_id = ?",
                (proposal.proposal_id,),
            )

    def get_prompt_override(self, agent_target: str) -> str | None:
        row = self._fetch_one(
            "SELECT prompt_override FROM prompt_runtime_overrides WHERE agent_target = ?",
            (agent_target,),
        )
        return row["prompt_override"] if row else None

    def save_pattern_writeback(
        self,
        symptom: str,
        source: str,
        skill_id: str,
        gpu_vendor: str | None,
        vulkan_version: str,
        hit_rate: float,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO pattern_writebacks(
                    symptom, source, skill_id, gpu_vendor, vulkan_version, hit_rate, under_review, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (symptom, source, skill_id, gpu_vendor, vulkan_version, hit_rate, _now()),
            )

    def get_pattern_writebacks(self) -> list[sqlite3.Row]:
        return self._fetch_all("SELECT * FROM pattern_writebacks ORDER BY updated_at DESC")

    def mark_pattern_under_review(self, symptom: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE pattern_writebacks SET under_review = 1 WHERE symptom = ?",
                (symptom,),
            )

    def get_pattern_writeback_status(self, symptom: str) -> dict[str, object] | None:
        row = self._fetch_one("SELECT * FROM pattern_writebacks WHERE symptom = ?", (symptom,))
        return dict(row) if row else None

    def _fetch_one(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(sql, params).fetchone()

    def _fetch_all(self, sql: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(sql, params).fetchall()


def _row_to_skill(row: sqlite3.Row) -> VulkanSkill:
    return VulkanSkill(
        skill_id=row["skill_id"],
        created_at=row["created_at"],
        last_used_at=row["last_used_at"],
        last_improved_at=row["last_improved_at"],
        name=row["name"],
        symptom=row["symptom"],
        domain=row["domain"],
        gpu_vendor=row["gpu_vendor"],
        vulkan_version_min=row["vulkan_version_min"],
        target_os=row["target_os"],
        fix_procedure=row["fix_procedure"],
        code_template=row["code_template"],
        cmake_snippet=row["cmake_snippet"],
        validation_step=row["validation_step"],
        times_applied=int(row["times_applied"]),
        times_successful=int(row["times_successful"]),
        confidence=row["confidence"],
        status=row["status"],
        retirement_reason=row["retirement_reason"],
    )


def _row_to_prompt_proposal(row: sqlite3.Row):
    from agents.self_improvement.prompt_refiner import PromptRefinementProposal

    return PromptRefinementProposal(
        proposal_id=row["proposal_id"],
        created_at=row["created_at"],
        agent_target=row["agent_target"],
        current_prompt_excerpt=row["current_prompt_excerpt"],
        proposed_change=row["proposed_change"],
        rationale=row["rationale"],
        evidence=json.loads(row["evidence_json"]),
        status=row["status"],
    )


def _version_gte(left: str, right: str) -> bool:
    return _version_tuple(left) >= _version_tuple(right)


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.replace("v", "").split(".") if part.isdigit())


def _now() -> str:
    return datetime.now(UTC).isoformat()
