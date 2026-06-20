from __future__ import annotations

from orchestrator.state import PlatformContext


def generate_vulkan_hpp_boilerplate(context: PlatformContext) -> str:
    vendor_note = ", ".join(f"{key}={value}" for key, value in sorted(context.target.quirk_profile.items()))
    return f"""#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

namespace vulkanmind {{

class DeviceAllocator {{
public:
    explicit DeviceAllocator(vk::Device device) : device_(device) {{
        VmaAllocatorCreateInfo create_info{{}};
        create_info.physicalDevice = static_cast<VkPhysicalDevice>(device_.getPhysicalDevice());
        create_info.device = static_cast<VkDevice>(device_);
        create_info.instance = static_cast<VkInstance>(device_.getInstance());
        if (vmaCreateAllocator(&create_info, &allocator_) != VK_SUCCESS) {{
            throw std::runtime_error("vmaCreateAllocator failed");
        }}
    }}

    DeviceAllocator(DeviceAllocator const&) = delete;
    DeviceAllocator& operator=(DeviceAllocator const&) = delete;

    ~DeviceAllocator() {{
        if (allocator_ != nullptr) {{
            vmaDestroyAllocator(allocator_);
        }}
    }}

    [[nodiscard]] VmaAllocator allocator() const noexcept {{ return allocator_; }}

private:
    vk::Device device_;
    VmaAllocator allocator_{{}};
}};

struct GeneratedContext {{
    std::string target_gpu_vendor{{"{context.target.gpu_vendor}"}};
    std::string target_gpu_model{{"{context.target.gpu_model}"}};
    std::string vulkan_version{{"{context.target.vulkan_version}"}};
    std::string quirk_profile{{"{vendor_note}"}};
}};

[[nodiscard]] std::expected<GeneratedContext, std::string> make_generated_context() noexcept {{
    return GeneratedContext{{}};
}}

}} // namespace vulkanmind
"""
