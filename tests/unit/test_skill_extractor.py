from agents.self_improvement.skill_extractor import SkillExtractor
from db.execution_traces import ExecutionTrace
from db.skill_writebacks import SkillStore


def _trace(outcome: str, iterations: int, pattern: str | None) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id="trace-1",
        session_id="session-1",
        timestamp="2026-01-01T00:00:00+00:00",
        task_type="debug",
        gpu_vendor="Qualcomm",
        gpu_model="Adreno",
        vulkan_version="1.3",
        target_os="Android",
        user_request="debug black screen",
        validation_passed=outcome == "success",
        outcome=outcome,
        iterations_required=iterations,
        bug_pattern_matched=pattern,
    )


def test_should_extract_only_nontrivial_successful_pattern_traces() -> None:
    extractor = SkillExtractor(None, SkillStore("data/test.sqlite3"))
    assert extractor.should_extract(_trace("success", 2, "BLACK_SCREEN")) is True
    assert extractor.should_extract(_trace("success", 1, "BLACK_SCREEN")) is False
    assert extractor.should_extract(_trace("failure", 3, "BLACK_SCREEN")) is False
