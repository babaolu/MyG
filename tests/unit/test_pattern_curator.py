from datetime import UTC, datetime

from agents.self_improvement.pattern_curator import PatternCurator
from db.execution_traces import ExecutionTraceStore
from db.skill_writebacks import SkillStore, VulkanSkill


def test_pattern_curator_promotes_high_success_skills(tmp_path) -> None:
    skill_store = SkillStore(str(tmp_path / "skills.sqlite3"))
    trace_store = ExecutionTraceStore(str(tmp_path / "traces.sqlite3"))
    skill = VulkanSkill(
        skill_id="skill-1",
        created_at=datetime.now(UTC).isoformat(),
        name="test skill",
        symptom="BLACK_SCREEN",
        domain="synchronization",
        gpu_vendor="Qualcomm",
        vulkan_version_min="1.3",
        target_os="Android",
        fix_procedure="fix",
        validation_step="validate",
        times_applied=3,
        times_successful=3,
        confidence="medium",
    )
    skill_store.save(skill)
    report = PatternCurator(skill_store, trace_store, type("PatternLibrary", (), {"BUG_PATTERNS": []})).run_weekly_audit()
    promoted = skill_store.get_by_id("skill-1")
    assert report.skills_promoted == 1
    assert promoted.confidence == "trusted"
