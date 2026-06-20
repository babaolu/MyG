from __future__ import annotations

from datetime import UTC, datetime

import structlog
from pydantic import BaseModel

from db.execution_traces import ExecutionTraceStore
from db.skill_writebacks import SkillStore


class CurationReport(BaseModel):
    run_at: str
    skills_reviewed: int
    skills_promoted: int
    skills_retired: int
    skills_flagged: int
    patterns_updated: int
    patterns_retired: int
    requires_human_review: list[str]


class PatternCurator:
    def __init__(
        self,
        skill_store: SkillStore,
        trace_store: ExecutionTraceStore,
        pattern_library,
    ):
        self.skill_store = skill_store
        self.trace_store = trace_store
        self.pattern_library = pattern_library

    def run_weekly_audit(self) -> CurationReport:
        logger = structlog.get_logger("vulkanmind.self_improvement.pattern_curator")
        report = CurationReport(
            run_at=datetime.now(UTC).isoformat(),
            skills_reviewed=0,
            skills_promoted=0,
            skills_retired=0,
            skills_flagged=0,
            patterns_updated=0,
            patterns_retired=0,
            requires_human_review=[],
        )
        active_skills = self.skill_store.get_all_active()
        report.skills_reviewed = len(active_skills)

        for skill in active_skills:
            if skill.confidence != "trusted" and skill.times_successful >= 3:
                skill.confidence = "trusted"
                skill.last_improved_at = report.run_at
                self.skill_store.save(skill)
                report.skills_promoted += 1
                logger.info("skill_promoted", skill_id=skill.skill_id, confidence=skill.confidence)

        for skill in self.skill_store.get_stale_skills(days_unused=90):
            success_rate = skill.times_successful / skill.times_applied if skill.times_applied else 0.0
            if skill.times_applied == 0:
                self.skill_store.retire(skill.skill_id, "unused_never_applied")
                report.skills_retired += 1
            elif success_rate < 0.4:
                self.skill_store.save_human_review(skill.skill_id, "skill", "success_rate_below_threshold")
                skill.status = "under_review"
                self.skill_store.save(skill)
                report.skills_flagged += 1
                report.requires_human_review.append(skill.skill_id)
            else:
                self.skill_store.retire(skill.skill_id, "unused_90_days")
                report.skills_retired += 1

        report.skills_flagged += self._flag_contradictions()

        for pattern in getattr(self.pattern_library, "BUG_PATTERNS", []):
            hit_rate = self.trace_store.get_pattern_hit_rate(pattern.symptom)
            metadata = getattr(self.pattern_library, "PATTERN_METADATA", {}).get(pattern.symptom, {})
            if hit_rate < 0.3 and metadata.get("source") == "skill_writeback":
                if hasattr(self.pattern_library, "mark_pattern_under_review"):
                    self.pattern_library.mark_pattern_under_review(pattern.symptom)
                if hasattr(self.skill_store, "mark_pattern_under_review"):
                    self.skill_store.mark_pattern_under_review(pattern.symptom)
                skill_id = metadata.get("skill_id")
                if skill_id:
                    self.skill_store.save_human_review(str(skill_id), "pattern", "hit_rate_below_threshold")
                    report.requires_human_review.append(str(skill_id))
                report.patterns_updated += 1

        self.skill_store.save_curation_report(report)
        logger.info("pattern_curation_completed", report=report.model_dump())
        return report

    def require_human_review(self, skill_ids: list[str]) -> None:
        for skill_id in skill_ids:
            skill = self.skill_store.get_by_id(skill_id)
            if skill is None:
                continue
            skill.status = "under_review"
            self.skill_store.save(skill)
            self.skill_store.save_human_review(skill_id, "skill", "human_review_required")

    def _flag_contradictions(self) -> int:
        grouped: dict[str, list] = {}
        for skill in self.skill_store.get_all_active():
            grouped.setdefault(skill.symptom, []).append(skill)
        flagged = 0
        for skills in grouped.values():
            if len(skills) < 2:
                continue
            sorted_skills = sorted(
                skills,
                key=lambda item: item.times_successful / item.times_applied if item.times_applied else 0.0,
                reverse=True,
            )
            for weaker in sorted_skills[1:]:
                if weaker.fix_procedure != sorted_skills[0].fix_procedure:
                    weaker.status = "under_review"
                    self.skill_store.save(weaker)
                    self.skill_store.save_human_review(weaker.skill_id, "skill", "contradictory_fix_procedure")
                    flagged += 1
        return flagged
