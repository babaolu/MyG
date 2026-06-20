from __future__ import annotations

import structlog
from pydantic import BaseModel

from orchestrator.state import BugHypothesis, PlatformContext


class BugPattern(BaseModel):
    symptom: str
    classification: str
    platform_filter: list[str] | None = None
    hypotheses: list[BugHypothesis]


BUG_PATTERNS: list[BugPattern] = [
    BugPattern(
        symptom="BLACK_SCREEN",
        classification="ISOLATABLE",
        hypotheses=[
            BugHypothesis(description="swapchain image not transitioned to PRESENT_SRC_KHR", probability=0.85, fix_template="Insert vk::PipelineBarrier before present.", validation_step="Verify image layout is PRESENT_SRC_KHR at vkQueuePresentKHR.", affected_platforms=["Android", "Qualcomm", "ARM"]),
            BugHypothesis(description="render pass store op DONT_CARE on final attachment", probability=0.80, fix_template="Use VK_STORE_OP_STORE for the presented attachment.", validation_step="Inspect VkAttachmentDescription::storeOp for the final color attachment."),
            BugHypothesis(description="viewport/scissor dynamic state not set", probability=0.75, fix_template="Call vkCmdSetViewport and vkCmdSetScissor before drawing.", validation_step="Capture command buffer and verify dynamic state commands."),
            BugHypothesis(description="wrong vertex winding order vs cull mode", probability=0.70, fix_template="Flip winding order or disable culling for validation.", validation_step="Render unculled triangles and compare primitive IDs."),
            BugHypothesis(description="NDC Y-axis flip from OpenGL habit in Vulkan context", probability=0.65, fix_template="Adjust projection matrix or viewport origin for Vulkan NDC.", validation_step="Check top-left vertex maps to expected framebuffer coordinate."),
            BugHypothesis(description="vertex buffer binding offset wrong", probability=0.60, fix_template="Set vkCmdBindVertexBuffers offsets to zero or correct byte offset.", validation_step="Validate binding offsets against buffer stride."),
            BugHypothesis(description="color blend write mask is zero", probability=0.55, fix_template="Enable VK_COLOR_COMPONENT_R_BIT|G_BIT|B_BIT|A_BIT.", validation_step="Inspect VkPipelineColorBlendAttachmentState::colorWriteMask."),
        ],
    ),
    BugPattern(
        symptom="GPU_HANG",
        classification="CROSS_SYSTEM",
        hypotheses=[
            BugHypothesis(description="unbounded loop in shader", probability=0.80, fix_template="Add loop bounds and validation counters.", validation_step="Run spirv-val and shader debug counters."),
            BugHypothesis(description="missing pipeline barrier for read/write hazard", probability=0.75, fix_template="Insert vkCmdPipelineBarrier with precise src/dst stages and layouts.", validation_step="Replay with validation layers and timeline semaphore checks."),
            BugHypothesis(description="out of bounds buffer access in shader", probability=0.70, fix_template="Clamp indices and validate descriptor ranges.", validation_step="Enable robust buffer access and shader instrumentation."),
        ],
    ),
    BugPattern(
        symptom="VALIDATION_SYNC",
        classification="ISOLATABLE",
        hypotheses=[
            BugHypothesis(description="srcStageMask too late - operation already started", probability=0.80, fix_template="Move producer stage into srcStageMask.", validation_step="Check producer command stage against barrier srcStageMask."),
            BugHypothesis(description="dstStageMask too early - consumer starts before ready", probability=0.75, fix_template="Extend dstStageMask to include consumer stage.", validation_step="Check consumer command stage against barrier dstStageMask."),
        ],
    ),
    BugPattern(
        symptom="VALIDATION_LAYOUT",
        classification="ISOLATABLE",
        hypotheses=[
            BugHypothesis(description="image used before layout transition", probability=0.85, fix_template="Transition image layout before first use.", validation_step="Compare image layout at each descriptor write."),
            BugHypothesis(description="wrong layout at render pass begin", probability=0.80, fix_template="Set correct initialLayout in VkAttachmentDescription.", validation_step="Inspect render pass begin info against attachment usage."),
        ],
    ),
]

PATTERN_METADATA: dict[str, dict[str, str]] = {}
_PATTERN_STORE: object | None = None


def configure_pattern_store(store) -> None:
    global _PATTERN_STORE
    _PATTERN_STORE = store


def match_patterns(text: str, platform_vendor: str | None = None) -> list[BugPattern]:
    lowered = text.lower()
    matched: list[BugPattern] = []
    for pattern in BUG_PATTERNS:
        symptom_hits = pattern.symptom.lower() in lowered
        keyword_hits = any(keyword.lower() in lowered for keyword in _pattern_keywords(pattern.symptom))
        if symptom_hits or keyword_hits:
            matched.append(pattern)
    return matched


def _pattern_keywords(symptom: str) -> list[str]:
    mapping = {
        "BLACK_SCREEN": ["present_src", "store op", "viewport", "scissor", "winding", "ndc", "blend", "write mask"],
        "GPU_HANG": ["gpu hang", "timeout", "loop", "barrier", "out of bounds"],
        "VALIDATION_SYNC": ["srcstagemask", "dststagemask", "pipeline barrier", "synchronization"],
        "VALIDATION_LAYOUT": ["layout", "image used", "render pass begin"],
    }
    return mapping.get(symptom, [symptom.lower()])


def get_all_symptoms() -> list[str]:
    return sorted({pattern.symptom for pattern in BUG_PATTERNS})


def write_back_fix(
    symptom: str,
    hypothesis: BugHypothesis,
    platform_context: PlatformContext,
    trace_id: str,
    skill_id: str,
) -> None:
    logger = structlog.get_logger("vulkanmind.self_improvement.pattern_writeback")
    pattern = next((item for item in BUG_PATTERNS if item.symptom == symptom), None)
    if pattern is None:
        pattern = BugPattern(
            symptom=symptom,
            classification=_infer_classification(hypothesis),
            platform_filter=[platform_context.target.gpu_vendor] if platform_context.target.gpu_vendor else None,
            hypotheses=[hypothesis],
        )
        BUG_PATTERNS.append(pattern)
    elif all(existing.description != hypothesis.description for existing in pattern.hypotheses):
        pattern.hypotheses.append(hypothesis)
    else:
        existing = next(existing for existing in pattern.hypotheses if existing.description == hypothesis.description)
        existing.probability = min(0.95, existing.probability + 0.05)
    PATTERN_METADATA[symptom] = {
        "source": "skill_writeback",
        "skill_id": skill_id,
        "trace_id": trace_id,
        "gpu_vendor": platform_context.target.gpu_vendor,
        "vulkan_version": platform_context.target.vulkan_version,
    }
    hit_rate = 1.0 if hypothesis.probability >= 0.75 else 0.0
    if _PATTERN_STORE is not None:
        _PATTERN_STORE.save_pattern_writeback(
            symptom=symptom,
            source="skill_writeback",
            skill_id=skill_id,
            gpu_vendor=platform_context.target.gpu_vendor,
            vulkan_version=platform_context.target.vulkan_version,
            hit_rate=hit_rate,
        )
    logger.info(
        "pattern_writeback_completed",
        symptom=symptom,
        skill_id=skill_id,
        trace_id=trace_id,
        hypothesis=hypothesis.description,
    )


def mark_pattern_under_review(symptom: str) -> None:
    metadata = PATTERN_METADATA.setdefault(symptom, {})
    metadata["under_review"] = "true"
    if _PATTERN_STORE is not None:
        _PATTERN_STORE.mark_pattern_under_review(symptom)


def _infer_classification(hypothesis: BugHypothesis) -> str:
    text = f"{hypothesis.description} {hypothesis.fix_template}".lower()
    if any(token in text for token in ["gpu hang", "driver reset", "thermal", "shader loop"]):
        return "CROSS_SYSTEM"
    return "ISOLATABLE"
