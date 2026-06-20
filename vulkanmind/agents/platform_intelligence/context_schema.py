from __future__ import annotations

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
    memory_heaps: list[dict] = Field(default_factory=list)
    quirk_profile: dict = Field(default_factory=dict)
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
