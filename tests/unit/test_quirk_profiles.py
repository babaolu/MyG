from agents.platform_intelligence.quirk_profiles import get_quirk_profile


def test_qualcomm_default_quirks() -> None:
    profile = get_quirk_profile("Qualcomm", "Adreno")
    assert profile["prefer_tiling_optimal"] is True
    assert profile["avoid_vkCmdClearImage"] is True


def test_unknown_vendor_defaults_to_qualcomm_profile() -> None:
    profile = get_quirk_profile("Unknown", "GPU")
    assert "prefer_tiling_optimal" in profile
