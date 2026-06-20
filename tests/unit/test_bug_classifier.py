from agents.debugger.classifier import classify_bug


def test_classify_black_screen_as_isolatable() -> None:
    result = classify_bug("Validation error: image used before layout transition", None, "black screen")
    assert result.classification == "ISOLATABLE"
    assert "VALIDATION_LAYOUT" in result.patterns


def test_classify_gpu_hang_as_cross_system() -> None:
    result = classify_bug("GPU hang timeout", None, "device lost")
    assert result.classification == "CROSS_SYSTEM"
    assert "GPU_HANG" in result.patterns
