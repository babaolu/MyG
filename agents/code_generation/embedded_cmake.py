from __future__ import annotations

from orchestrator.state import PlatformContext


def generate_cmake_snippet(context: PlatformContext) -> str:
    return """cmake_minimum_required(VERSION 3.25)
project(VulkanMindGenerated LANGUAGES CXX)

find_package(Vulkan REQUIRED)

add_library(vulkanmind_generated STATIC src/main.cpp)
target_compile_features(vulkanmind_generated PRIVATE cxx_std_20)
target_link_libraries(vulkanmind_generated PRIVATE Vulkan::Vulkan)
target_compile_options(vulkanmind_generated PRIVATE -Os -Wall -Wextra)
"""
