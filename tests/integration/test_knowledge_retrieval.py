from agents.knowledge_retrieval.retrieval.query_builder import build_query
from orchestrator.state import (
    HostContext,
    PlatformContext,
    TargetContext,
    ThermalContext,
    ToolchainContext,
)


def _context() -> PlatformContext:
    return PlatformContext(
        host=HostContext(os="Linux", arch="x86_64", compiler="clang++", compiler_version="18", auto_detected=True),
        target=TargetContext(os="Android", arch="arm64-v8a", gpu_vendor="Qualcomm", gpu_model="Adreno", vulkan_version="1.3", source="user_declared"),
        toolchain=ToolchainContext(is_cross_compile=True),
        thermal=ThermalContext(mode="NORMAL", available_cores=8, cpu_load_percent=10.0, available_ram_percent=70.0, max_parallel_workers=4),
        runtime_validation_available=False,
    )


def test_build_query_injects_platform_filters() -> None:
    query = build_query("pipeline barrier", _context(), topic_hint="synchronization")
    assert query.limit == 8
    assert query.query_filter.must
    assert query.query_filter.should
