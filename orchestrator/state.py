from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field


class HostContext(BaseModel):
    os: str
    arch: str
    compiler: str
    compiler_version: str
    auto_detected: bool = True


class TargetContext(BaseModel):
    os: str
    arch: str
    gpu_vendor: str
    gpu_model: str
    driver_version: str | None = None
    vulkan_version: str
    supported_extensions: list[str] = Field(default_factory=list)
    unsupported_extensions: list[str] = Field(default_factory=list)
    memory_heaps: list[dict[str, Any]] = Field(default_factory=list)
    quirk_profile: dict[str, Any] = Field(default_factory=dict)
    auto_detected: bool = False
    source: str
    model_config = ConfigDict(extra="allow")


class ToolchainContext(BaseModel):
    is_cross_compile: bool
    cross_compiler: str | None = None
    ndk_version: str | None = None
    ndk_path: str | None = None
    sysroot: str | None = None
    cmake_toolchain_file: str | None = None


class ThermalContext(BaseModel):
    mode: str
    available_cores: int
    cpu_load_percent: float
    cpu_temp_celsius: float | None = None
    available_ram_percent: float
    battery_percent: float | None = None
    max_parallel_workers: int


class PlatformContext(BaseModel):
    host: HostContext
    target: TargetContext
    toolchain: ToolchainContext
    thermal: ThermalContext
    runtime_validation_available: bool


class BugHypothesis(BaseModel):
    description: str
    probability: float = Field(ge=0.0, le=1.0)
    fix_template: str
    validation_step: str
    affected_platforms: list[str] | None = None


class KnowledgeChunk(BaseModel):
    id: str
    text: str
    source_title: str
    source_type: str
    platform_tags: list[str] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)
    vulkan_version: str | None = None
    confidence: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class BuildCandidate(BaseModel):
    id: str
    source_file: str
    cmake_preset: str
    status: str = "queued"
    created_at: float


class BugClassification(BaseModel):
    classification: str
    confidence: float
    rationale: str
    patterns: list[str] = Field(default_factory=list)


class SessionMemory(BaseModel):
    relevant_skills: list[Any] = Field(default_factory=list)
    recent_fixes: list[Any] = Field(default_factory=list)
    platform_hit_rate: dict[str, float] = Field(default_factory=dict)
    injected_at: str | None = None


class VulkanMindState(BaseModel):
    session_id: str
    task_type: Literal[
        "code_generation",
        "debug",
        "knowledge_query",
        "platform_detect",
        "self_update",
        "unknown",
    ]
    platform_context: PlatformContext | None = None
    user_request: str
    attached_files: list[str] = Field(default_factory=list)
    validation_output: str | None = None
    build_log: str | None = None
    generated_code: str | None = None
    cmake_snippet: str | None = None
    validation_passed: bool = False
    bug_classification: BugClassification | None = None
    bug_hypotheses: list[BugHypothesis] = Field(default_factory=list)
    active_fix: str | None = None
    build_queue: list[BuildCandidate] = Field(default_factory=list)
    hardware_governor_mode: str | None = None
    retrieved_knowledge: list[KnowledgeChunk] = Field(default_factory=list)
    messages: list[BaseMessage] = Field(default_factory=list)
    agent_trace: list[str] = Field(default_factory=list)
    error: str | None = None
    self_improvement_phase: Literal["start", "end"] = "start"
    session_memory: SessionMemory | None = None
    skill_extracted: bool = False
    skill_id: str | None = None
    trace_id: str | None = None
    improvement_context: str | None = None
    topic_hint: str | None = None
    graphify_snapshot: str | None = None
    target_platform_declared: dict[str, Any] | None = None
    target_platform: dict[str, Any] | None = None
    adb_connected: bool = False
    # Optional collaborators injected by the runtime (memory injector, Graphify
    # reader, etc.). Pydantic v2 chokes on arbitrary objects without arbitrary_types_allowed.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")
    memory_injector: Any | None = None
    graphify_reader: Any | None = None
    llm_client: Any | None = None


def coerce_state(
    state: VulkanMindState | dict[str, Any],
) -> VulkanMindState:
    """Return a typed ``VulkanMindState`` for whatever the caller passed in.

    Graph nodes historically accepted ``state: dict`` because LangGraph emits
    raw mappings from state reducers. Wrapping that conversion in one helper
    means every node can start with the same two-line pattern, and tests can
    use ``VulkanMindState`` directly without losing the run-time path.

    ``session_id`` and ``task_type`` are required by the model schema. When
    they are missing (e.g. minimal dict-based unit-test fixtures) we default
    them so node logic can still run; callers that need real persistence
    always populate them explicitly.
    """
    if isinstance(state, VulkanMindState):
        return state
    if isinstance(state, dict):
        if "session_id" not in state:
            state = {**state, "session_id": state.get("session_id") or "test-session"}
        if "task_type" not in state:
            state = {**state, "task_type": state.get("task_type") or "unknown"}
    return VulkanMindState.model_validate(state)
