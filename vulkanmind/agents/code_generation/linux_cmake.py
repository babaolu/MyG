from __future__ import annotations

from orchestrator.state import PlatformContext


def generate_cmake_snippet(context: PlatformContext) -> str:
    return """cmake_minimum_required(VERSION 3.25)
project(VulkanMindGenerated LANGUAGES CXX)

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG QUIET)
find_package(glm CONFIG QUIET)

add_executable(vulkanmind_generated src/main.cpp)
target_compile_features(vulkanmind_generated PRIVATE cxx_std_20)
target_link_libraries(vulkanmind_generated PRIVATE Vulkan::Vulkan glfw glm::glm)
target_compile_options(vulkanmind_generated PRIVATE -Wall -Wextra -Wpedantic)
"""
