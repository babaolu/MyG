from __future__ import annotations

from orchestrator.state import BugClassification

from .pattern_library import match_patterns


def classify_bug(validation_output: str | None, build_log: str | None, user_description: str | None) -> BugClassification:
    text = "\n".join(part for part in [validation_output, build_log, user_description] if part)
    patterns = match_patterns(text)
    if not patterns:
        classification = "ISOLATABLE" if _looks_single_stage(text) else "CROSS_SYSTEM"
        confidence = 0.55
        rationale = "No seeded pattern matched; classification based on diagnostic scope."
    else:
        classifications = [pattern.classification for pattern in patterns]
        classification = "CROSS_SYSTEM" if "CROSS_SYSTEM" in classifications else classifications[0]
        confidence = max((max(pattern.hypotheses, key=lambda item: item.probability).probability for pattern in patterns), default=0.5)
        rationale = "Matched seeded Vulkan bug patterns."
    return BugClassification(
        classification=classification,
        confidence=round(confidence, 2),
        rationale=rationale,
        patterns=[pattern.symptom for pattern in patterns],
    )


def _looks_single_stage(text: str) -> bool:
    lowered = text.lower()
    single_stage_markers = ["srcstagemask", "dststagemask", "layout", "viewport", "scissor", "storeop"]
    cross_markers = ["gpu hang", "device lost", "timeout", "thermal", "driver reset"]
    return any(marker in lowered for marker in single_stage_markers) and not any(marker in lowered for marker in cross_markers)
