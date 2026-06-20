from __future__ import annotations

__all__ = [
    "BuildQueueStore",
    "BugHistoryStore",
    "ExecutionTrace",
    "ExecutionTraceStore",
    "SessionStore",
    "SkillStore",
    "VulkanSkill",
]


def __getattr__(name: str):
    if name == "BuildQueueStore":
        from .build_queue_store import BuildQueueStore

        return BuildQueueStore
    if name == "BugHistoryStore":
        from .bug_history import BugHistoryStore

        return BugHistoryStore
    if name == "ExecutionTrace":
        from .execution_traces import ExecutionTrace

        return ExecutionTrace
    if name == "ExecutionTraceStore":
        from .execution_traces import ExecutionTraceStore

        return ExecutionTraceStore
    if name == "SessionStore":
        from .session_store import SessionStore

        return SessionStore
    if name == "SkillStore":
        from .skill_writebacks import SkillStore

        return SkillStore
    if name == "VulkanSkill":
        from .skill_writebacks import VulkanSkill

        return VulkanSkill
    raise AttributeError(name)
