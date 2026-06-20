from agents.platform_intelligence.context_schema import HostContext
from agents.platform_intelligence.detector import detect_target_platform, determine_cross_compile


def test_detect_target_platform_user_declared() -> None:
    target = detect_target_platform(
        {
            "os": "Android",
            "arch": "arm64-v8a",
            "gpu_vendor": "Qualcomm",
            "gpu_model": "Adreno 740",
            "vulkan_version": "1.3",
            "supported_extensions": ["VK_KHR_timeline_semaphore"],
        },
        adb_connected=False,
    )
    assert target.source == "user_declared"
    assert target.gpu_vendor == "Qualcomm"
    assert target.vulkan_version == "1.3"
    assert "prefer_tiling_optimal" in target.quirk_profile


def test_determine_cross_compile_linux_to_android() -> None:
    host = HostContext(os="Linux", arch="x86_64", compiler="clang++", compiler_version="18", auto_detected=True)
    target = detect_target_platform({"os": "Android", "arch": "arm64-v8a", "gpu_vendor": "ARM", "gpu_model": "Mali", "vulkan_version": "1.2"}, False)
    toolchain = determine_cross_compile(host, target)
    assert toolchain.is_cross_compile is True
    assert toolchain.cmake_toolchain_file is None or toolchain.cmake_toolchain_file.endswith("android.toolchain.cmake")
