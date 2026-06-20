from __future__ import annotations

from typing import Any

QUIRK_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "Qualcomm": {
        "default": {
            "prefer_tiling_optimal": True,
            "avoid_vkCmdClearImage": True,
            "prefer_load_op_clear": True,
            "ubwc_compression": "enable",
            "prefer_subpass_dependencies": True,
            "mediump_precision_warning": True,
        }
    },
    "ARM": {
        "default": {
            "tbdr_architecture": True,
            "avoid_mid_frame_readback": True,
            "prefer_interleaved_vertex_attributes": True,
            "use_dont_care_for_transient": True,
            "avoid_partial_clears": True,
        }
    },
    "PowerVR": {
        "default": {
            "tbdr_architecture": True,
            "tile_based_optimizations": True,
        }
    },
    "NVIDIA": {
        "default": {
            "discrete_memory": True,
            "explicit_staging_buffers": True,
            "prefer_device_local": True,
        },
        "Turing+": {
            "mesh_shaders_available": True,
            "ray_tracing_available": True,
        },
    },
    "AMD": {
        "default": {
            "rdna_wave32_default": True,
            "prefer_compute_skinning": True,
        }
    },
    "Intel": {
        "default": {
            "prefer_integrated_memory": True,
            "avoid_large_transient_allocations": True,
        }
    },
}


def get_quirk_profile(gpu_vendor: str, gpu_model: str) -> dict[str, Any]:
    vendor_key = next(
        (vendor for vendor in QUIRK_PROFILES if vendor.lower() == gpu_vendor.lower()),
        None,
    )
    if vendor_key is None:
        return dict(QUIRK_PROFILES["Qualcomm"]["default"])
    merged: dict[str, Any] = {}
    merged.update(QUIRK_PROFILES[vendor_key]["default"])
    pattern_key = next(
        (key for key in QUIRK_PROFILES[vendor_key] if key != "default" and key.lower() in gpu_model.lower()),
        None,
    )
    if pattern_key:
        merged.update(QUIRK_PROFILES[vendor_key][pattern_key])
    return merged
