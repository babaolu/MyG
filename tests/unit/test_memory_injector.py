from datetime import UTC, datetime

from agents.self_improvement.memory_injector import MemoryInjector
from db.execution_traces import ExecutionTrace, ExecutionTraceStore
from db.skill_writebacks import SkillStore, VulkanSkill
from orchestrator.state import (
    HostContext,
    PlatformContext,
    TargetContext,
    ThermalContext,
    ToolchainContext,
)


def _context() -> PlatformContext:
    return PlatformContext(
        host=HostContext(os="Linux", arch="x86_64", compiler="clang++", compiler_version="18"),
        target=TargetContext(os="Android", arch="arm64", gpu_vendor="Qualcomm", gpu_model="Adreno", vulkan_version="1.3", source="user_declared"),
        toolchain=ToolchainContext(is_cross_compile=True),
        thermal=ThermalContext(mode="NORMAL", available_cores=8, cpu_load_percent=10.0, available_ram_percent=80.0, max_parallel_workers=4),
        runtime_validation_available=False,
    )


def test_memory_injector_filters_skills_and_recent_fixes_by_platform(tmp_path) -> None:
    store = SkillStore(str(tmp_path / "skills.sqlite3"))
    trace_store = ExecutionTraceStore(str(tmp_path / "traces.sqlite3"))
    store.save(
        VulkanSkill(
            skill_id="skill-adreno",
            created_at=datetime.now(UTC).isoformat(),
            name="adreno fix",
            symptom="BLACK_SCREEN",
            domain="synchronization",
            gpu_vendor="Qualcomm",
            vulkan_version_min="1.3",
            target_os="Android",
            fix_procedure="fix",
            validation_step="validate",
            times_successful=3,
            confidence="trusted",
        )
    )
    store.save(
        VulkanSkill(
            skill_id="skill-nvidia",
            created_at=datetime.now(UTC).isoformat(),
            name="nvidia fix",
            symptom="BLACK_SCREEN",
            domain="synchronization",
            gpu_vendor="NVIDIA",
            vulkan_version_min="1.3",
            target_os="Linux",
            fix_procedure="fix",
            validation_step="validate",
            times_successful=3,
            confidence="trusted",
        )
    )
    trace_store.record(
        ExecutionTrace(
            trace_id="trace-1",
            session_id="session-1",
            timestamp=datetime.now(UTC).isoformat(),
            task_type="debug",
            gpu_vendor="Qualcomm",
            gpu_model="Adreno",
            vulkan_version="1.3",
            target_os="Android",
            user_request="debug",
            validation_passed=True,
            outcome="success",
            iterations_required=2,
            bug_pattern_matched="BLACK_SCREEN",
        )
    )
    memory = MemoryInjector(store, trace_store).build_session_memory(_context(), "debug")
    assert [skill.skill_id for skill in memory.relevant_skills] == ["skill-adreno"]
    assert [trace.trace_id for trace in memory.recent_fixes] == ["trace-1"]
