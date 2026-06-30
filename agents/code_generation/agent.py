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
#include <vk_mem_allocate.h>

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


def generate_triangle_code(context: PlatformContext) -> str:
    return """#include <array>
#include <cstdint>
#include <expected>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>
#include <vk_mem_allocate.h>

namespace vulkanmind {

// Vertex data for triangle
struct Vertex {
    float x, y, z;
    float r, g, b;
};

class SimpleTriangleApp {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Triangle", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetWindowSizeCallback(window_, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int /*width*/, int /*height*/) {
        auto app = static_cast<SimpleTriangleApp*>(glfwGetWindowUserPointer(window));
        if (app) app->framebufferResized_ = true;
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createVertexBuffer();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            drawFrame();
        }
        device_.waitIdle();
    }

    void cleanup() {
        cleanupSyncObjects();
        device_.destroyBuffer(vertexBuffer_);
        device_.freeMemory(vertexBufferMemory_);
        device_.freeCommandBuffers(commandPool_, commandBuffers_);
        device_.destroyCommandPool(commandPool_);
        for (auto framebuffer : swapchainFramebuffers_) {
            device_.destroyFramebuffer(framebuffer);
        }
        device_.destroyPipeline(graphicsPipeline_);
        device_.destroyPipelineLayout(pipelineLayout_);
        device_.destroyRenderPass(renderPass_);
        for (auto imageView : swapchainImageViews_) {
            device_.destroyImageView(imageView);
        }
        device_.destroySwapchainKHR(swapchain_);
        device_.destroyDevice();
        instance_.destroySurfaceKHR(surface_);
        instance_.destroy();
        glfwDestroyWindow(window_);
        glfwTerminate();
    }

    void drawFrame() {
        uint32_t imageIndex;
        vk::Result result = device_.acquireNextImageKHR(swapchain_, UINT64_MAX,
            imageAvailableSemaphores_[currentFrame_], nullptr, &imageIndex);

        if (result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapchain();
            return;
        }

        vk::SubmitInfo submitInfo{};
        vk::Semaphore waitSemaphores[] = {imageAvailableSemaphores_[currentFrame_]};
        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        submitInfo.setWaitSemaphores(waitSemaphores)
                 .setWaitDstStageMask(waitStages)
                 .setCommandBuffers(commandBuffers_[imageIndex])
                 .setSignalSemaphores(renderFinishedSemaphores_[currentFrame_]);

        device_.resetFences(1, &inFlightFences_[currentFrame_]);
        graphicsQueue_.submit(submitInfo, inFlightFences_[currentFrame_]);

        vk::PresentInfoKHR presentInfo{};
        presentInfo.setWaitSemaphores(renderFinishedSemaphores_[currentFrame_])
                  .setSwapchains(swapchain_)
                  .setImageIndices(imageIndex);

        result = graphicsQueue_.presentKHR(presentInfo);
        if (result == vk::Result::eErrorOutOfDateKHR || framebufferResized_) {
            framebufferResized_ = false;
            recreateSwapchain();
        }

        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("shaders/triangle.vert.spv");
        auto fragShaderCode = readFile("shaders/triangle.frag.spv");

        vk::ShaderModuleCreateInfo vertCreateInfo{};
        vertCreateInfo.codeSize = vertShaderCode.size();
        vertCreateInfo.pCode = reinterpret_cast<const uint32_t*>(vertShaderCode.data());
        vk::ShaderModule vertModule = device_.createShaderModule(vertCreateInfo);

        vk::ShaderModuleCreateInfo fragCreateInfo{};
        fragCreateInfo.codeSize = fragShaderCode.size();
        fragCreateInfo.pCode = reinterpret_cast<const uint32_t*>(fragShaderCode.data());
        vk::ShaderModule fragModule = device_.createShaderModule(fragCreateInfo);

        vk::PipelineShaderStageCreateInfo vertStageInfo{};
        vertStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
        vertStageInfo.module = vertModule;
        vertStageInfo.pName = "main";

        vk::PipelineShaderStageCreateInfo fragStageInfo{};
        fragStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
        fragStageInfo.module = fragModule;
        fragStageInfo.pName = "main";

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertStageInfo, fragStageInfo};

        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;

        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions = {{
            {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, x)},
            {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, r)}
        }};

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;

        vk::Viewport viewport{0.0f, 0.0f, static_cast<float>(swapchainExtent_.width),
                              static_cast<float>(swapchainExtent_.height), 0.0f, 1.0f};
        vk::Rect2D scissor{{0, 0}, swapchainExtent_};

        vk::PipelineViewportStateCreateInfo viewportState{};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.pScissors = &scissor;

        vk::PipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = vk::PolygonMode::eFill;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = vk::CullModeFlagBits::eBack;
        rasterizer.frontFace = vk::FrontFace::eClockwise;

        vk::PipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR |
                                               vk::ColorComponentFlagBits::eG |
                                               vk::ColorComponentFlagBits::eB |
                                               vk::ColorComponentFlagBits::eA;

        vk::PipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayout_ = device_.createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout_;
        pipelineInfo.renderPass = renderPass_;
        pipelineInfo.subpass = 0;

        graphicsPipeline_ = device_.createGraphicsPipelines(nullptr, pipelineInfo).value[0];

        device_.destroyShaderModule(vertModule);
        device_.destroyShaderModule(fragModule);
    }

    std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("failed to open file: " + filename);
        size_t size = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(size);
        file.seekg(0);
        file.read(buffer.data(), size);
        return buffer;
    }

    static constexpr int WIDTH = 800;
    static constexpr int HEIGHT = 600;
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    GLFWwindow* window_ = nullptr;
    vk::Instance instance_;
    vk::SurfaceKHR surface_;
    vk::PhysicalDevice physicalDevice_;
    vk::Device device_;
    vk::Queue graphicsQueue_;
    vk::Queue presentQueue_;
    vk::SwapchainKHR swapchain_;
    std::vector<vk::Image> swapchainImages_;
    vk::Format swapchainImageFormat_;
    vk::Extent2D swapchainExtent_;
    std::vector<vk::ImageView> swapchainImageViews_;
    vk::RenderPass renderPass_;
    vk::PipelineLayout pipelineLayout_;
    vk::Pipeline graphicsPipeline_;
    std::vector<vk::Framebuffer> swapchainFramebuffers_;
    vk::CommandPool commandPool_;
    vk::Buffer vertexBuffer_;
    vk::DeviceMemory vertexBufferMemory_;
    std::vector<vk::CommandBuffer> commandBuffers_;
    std::vector<vk::Semaphore> imageAvailableSemaphores_;
    std::vector<vk::Semaphore> renderFinishedSemaphores_;
    std::vector<vk::Fence> inFlightFences_;
    bool framebufferResized_ = false;
    int currentFrame_ = 0;
};

int main() {
    vulkanmind::SimpleTriangleApp app;
    app.run();
    return 0;
}
"""


def generate_textured_shape_code(context: PlatformContext, shape: str) -> str:
    vendor = context.target.gpu_vendor
    return f"""#include <array>
#include <cstdint>
#include <expected>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>
#include <vk_mem_allocate.h>

namespace vulkanmind {{

// Vertex with UV for textured shape (target: {vendor})
struct Vertex {{
    float x, y, z;
    float u, v;
}};

// Generate UV vertices for a textured {shape}
static std::vector<Vertex> generateTexturedVertices(int segments = 64) {{
    std::vector<Vertex> vertices;
    vertices.reserve(segments + 1);
    float angleStep = 2.0f * 3.14159265f / segments;
    for (int i = 0; i <= segments; ++i) {{
        float angle = i * angleStep;
        float sinA = sin(angle);
        float cosA = cos(angle);
        vertices.push_back({{(cosA * 0.25f, sinA * 0.25f, 0.0f, 0.5f + cosA * 0.5f, 0.5f + sinA * 0.5f)}});
    }}
    return vertices;
}}

class TexturedShapeApp {{
public:
    void run() {{
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }}

private:
    void initWindow() {{
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Textured {shape.title()}", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetWindowSizeCallback(window_, framebufferResizeCallback);
    }}

    static void framebufferResizeCallback(GLFWwindow* window, int /*width*/, int /*height*/) {{
        auto app = static_cast<TexturedShapeApp*>(glfwGetWindowUserPointer(window));
        if (app) app->framebufferResized_ = true;
    }}

    void initVulkan() {{
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createVertexBuffer();
        createCommandBuffers();
        createSyncObjects();
    }}

    std::vector<char> readFile(const std::string& filename) {{
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("failed to open file: " + filename);
        size_t size = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(size);
        file.seekg(0);
        file.read(buffer.data(), size);
        return buffer;
    }}

    void mainLoop() {{
        while (!glfwWindowShouldClose(window_)) {{
            glfwPollEvents();
            drawFrame();
        }}
        device_.waitIdle();
    }}

    void cleanup() {{
        cleanupSyncObjects();
        device_.destroyBuffer(vertexBuffer_);
        device_.freeMemory(vertexBufferMemory_);
        device_.freeCommandBuffers(commandPool_, commandBuffers_);
        device_.destroyCommandPool(commandPool_);
        for (auto framebuffer : swapchainFramebuffers_) {{
            device_.destroyFramebuffer(framebuffer);
        }}
        device_.destroyPipeline(graphicsPipeline_);
        device_.destroyPipelineLayout(pipelineLayout_);
        device_.destroyRenderPass(renderPass_);
        for (auto imageView : swapchainImageViews_) {{
            device_.destroyImageView(imageView);
        }}
        device_.destroySwapchainKHR(swapchain_);
        device_.destroyDevice();
        instance_.destroySurfaceKHR(surface_);
        instance_.destroy();
        glfwDestroyWindow(window_);
        glfwTerminate();
    }}

    void drawFrame() {{
        uint32_t imageIndex;
        vk::Result result = device_.acquireNextImageKHR(swapchain_, UINT64_MAX,
            imageAvailableSemaphores_[currentFrame_], nullptr, &imageIndex);

        if (result == vk::Result::eErrorOutOfDateKHR) {{
            recreateSwapchain();
            return;
        }}

        vk::SubmitInfo submitInfo{{}};
        vk::Semaphore waitSemaphores[] = {{imageAvailableSemaphores_[currentFrame_]}};
        vk::PipelineStageFlags waitStages[] = {{vk::PipelineStageFlagBits::eColorAttachmentOutput}};
        submitInfo.setWaitSemaphores(waitSemaphores)
                 .setWaitDstStageMask(waitStages)
                 .setCommandBuffers(commandBuffers_[imageIndex])
                 .setSignalSemaphores(renderFinishedSemaphores_[currentFrame_]);

        device_.resetFences(1, &inFlightFences_[currentFrame_]);
        graphicsQueue_.submit(submitInfo, inFlightFences_[currentFrame_]);

        vk::PresentInfoKHR presentInfo{{}};
        presentInfo.setWaitSemaphores(renderFinishedSemaphores_[currentFrame_])
                  .setSwapchains(swapchain_)
                  .setImageIndices(imageIndex);

        result = graphicsQueue_.presentKHR(presentInfo);
        if (result == vk::Result::eErrorOutOfDateKHR || framebufferResized_) {{
            framebufferResized_ = false;
            recreateSwapchain();
        }}

        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
    }}

    void createGraphicsPipeline() {{
        auto vertShaderCode = readFile("shaders/textured.vert.spv");
        auto fragShaderCode = readFile("shaders/textured.frag.spv");

        vk::ShaderModuleCreateInfo vertCreateInfo{{}};
        vertCreateInfo.codeSize = vertShaderCode.size();
        vertCreateInfo.pCode = reinterpret_cast<const uint32_t*>(vertShaderCode.data());
        vk::ShaderModule vertModule = device_.createShaderModule(vertCreateInfo);

        vk::ShaderModuleCreateInfo fragCreateInfo{{}};
        fragCreateInfo.codeSize = fragShaderCode.size();
        fragCreateInfo.pCode = reinterpret_cast<const uint32_t*>(fragShaderCode.data());
        vk::ShaderModule fragModule = device_.createShaderModule(fragCreateInfo);

        vk::PipelineShaderStageCreateInfo vertStageInfo{{}};
        vertStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
        vertStageInfo.module = vertModule;
        vertStageInfo.pName = "main";

        vk::PipelineShaderStageCreateInfo fragStageInfo{{}};
        fragStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
        fragStageInfo.module = fragModule;
        fragStageInfo.pName = "main";

        vk::PipelineShaderStageCreateInfo shaderStages[] = {{vertStageInfo, fragStageInfo}};

        vk::VertexInputBindingDescription bindingDescription{{}};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;

        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions = {{
            {{0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, x)}},
            {{1, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, u)}}
        }};

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{{}};
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{{}};
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleFan;

        vk::Viewport viewport{{0.0f, 0.0f, static_cast<float>(swapchainExtent_.width),
                              static_cast<float>(swapchainExtent_.height), 0.0f, 1.0f}};
        vk::Rect2D scissor{{{{0, 0}}, swapchainExtent_}};

        vk::PipelineViewportStateCreateInfo viewportState{{}};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.pScissors = &scissor;

        vk::PipelineRasterizationStateCreateInfo rasterizer{{}};
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = vk::PolygonMode::eFill;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = vk::CullModeFlagBits::eBack;
        rasterizer.frontFace = vk::FrontFace::eClockwise;

        vk::PipelineMultisampleStateCreateInfo multisampling{{}};
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{{}};
        colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR |
                                               vk::ColorComponentFlagBits::eG |
                                               vk::ColorComponentFlagBits::eB |
                                               vk::ColorComponentFlagBits::eA;

        vk::PipelineColorBlendStateCreateInfo colorBlending{{}};
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{{}};
        pipelineLayout_ = device_.createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{{}};
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout_;
        pipelineInfo.renderPass = renderPass_;
        pipelineInfo.subpass = 0;

        graphicsPipeline_ = device_.createGraphicsPipelines(nullptr, pipelineInfo).value[0];

        device_.destroyShaderModule(vertModule);
        device_.destroyShaderModule(fragModule);
    }}

    static constexpr int WIDTH = 800;
    static constexpr int HEIGHT = 600;
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    GLFWwindow* window_ = nullptr;
    vk::Instance instance_;
    vk::SurfaceKHR surface_;
    vk::PhysicalDevice physicalDevice_;
    vk::Device device_;
    vk::Queue graphicsQueue_;
    vk::Queue presentQueue_;
    vk::SwapchainKHR swapchain_;
    std::vector<vk::Image> swapchainImages_;
    vk::Format swapchainImageFormat_;
    vk::Extent2D swapchainExtent_;
    std::vector<vk::ImageView> swapchainImageViews_;
    vk::RenderPass renderPass_;
    vk::PipelineLayout pipelineLayout_;
    vk::Pipeline graphicsPipeline_;
    std::vector<vk::Framebuffer> swapchainFramebuffers_;
    vk::CommandPool commandPool_;
    vk::Buffer vertexBuffer_;
    vk::DeviceMemory vertexBufferMemory_;
    std::vector<vk::CommandBuffer> commandBuffers_;
    std::vector<vk::Semaphore> imageAvailableSemaphores_;
    std::vector<vk::Semaphore> renderFinishedSemaphores_;
    std::vector<vk::Fence> inFlightFences_;
    bool framebufferResized_ = false;
    int currentFrame_ = 0;
}};

int main() {{
    vulkanmind::TexturedShapeApp app;
    app.run();
    return 0;
}}
"""


def generate_swapchain_code(context: PlatformContext) -> str:
    vendor = context.target.gpu_vendor
    return f"""#include <vulkan/vulkan.hpp>

namespace vulkanmind {{

// Swapchain management utilities for target GPU: {vendor}
class SwapchainManager {{
public:
    struct SwapchainImages {{
        vk::SwapchainKHR swapchain;
        std::vector<vk::Image> images;
        vk::Format format;
        vk::Extent2D extent;
    }};

    SwapchainImages createSwapchain(vk::PhysicalDevice physicalDevice,
                                   vk::Device device,
                                   vk::SurfaceKHR surface,
                                   uint32_t graphicsQueueFamily,
                                   uint32_t presentQueueFamily) {{
        SwapchainImages result{{}};

        // Query swapchain support
        vk::SurfaceCapabilitiesKHR capabilities = device.getSurfaceCapabilitiesKHR(surface);
        std::vector<vk::SurfaceFormatKHR> formats = device.getSurfaceFormatsKHR(surface);
        std::vector<vk::PresentModeKHR> presentModes = device.getSurfacePresentModesKHR(surface);

        // Choose format
        vk::SurfaceFormatKHR chosenFormat = formats[0];
        for (const auto& fmt : formats) {{
            if (fmt.format == vk::Format::eB8G8R8A8Srgb &&
                fmt.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {{
                chosenFormat = fmt;
                break;
            }}
        }}
        result.format = chosenFormat.format;
        result.extent = capabilities.currentExtent;

        // Choose present mode
        vk::PresentModeKHR presentMode = vk::PresentModeKHR::eFifo;
        for (const auto& pm : presentModes) {{
            if (pm == vk::PresentModeKHR::eMailbox) {{
                presentMode = pm;
                break;
            }}
        }}

        uint32_t imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {{
            imageCount = capabilities.maxImageCount;
        }}

        vk::SwapchainCreateInfoKHR createInfo{{}};
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = result.format;
        createInfo.imageColorSpace = chosenFormat.colorSpace;
        createInfo.imageExtent = result.extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

        uint32_t queueFamilyIndices[] = {{graphicsQueueFamily, presentQueueFamily}};
        if (graphicsQueueFamily != presentQueueFamily) {{
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }} else {{
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        }}

        createInfo.preTransform = capabilities.currentTransform;
        createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = nullptr;

        result.swapchain = device.createSwapchainKHR(createInfo);
        result.images = device.getSwapchainImagesKHR(result.swapchain);

        return result;
    }}
}};

}} // namespace vulkanmind
"""


def generate_geometry_code(context: PlatformContext, shape: str) -> str:
    """Generate vertex/index data for arbitrary geometry shapes."""
    if shape == "cube":
        verts = """// Cube vertices (positions + colors)
static const std::array<Vertex, 8> cubeVertices = {{
    {{-0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f}},  // 0: red
    {{ 0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 0.0f}},  // 1: green
    {{ 0.5f,  0.5f, -0.5f, 0.0f, 0.0f, 1.0f}},  // 2: blue
    {{-0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 0.0f}},  // 3: yellow
    {{-0.5f, -0.5f,  0.5f, 1.0f, 0.0f, 1.0f}},  // 4: magenta
    {{ 0.5f, -0.5f,  0.5f, 0.0f, 1.0f, 1.0f}},  // 5: cyan
    {{ 0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f}},  // 6: white
    {{-0.5f,  0.5f,  0.5f, 0.5f, 0.5f, 0.5f}},  // 7: gray
}};

// Cube indices (12 triangles)
static const std::array<uint32_t, 36> cubeIndices = {{
    0, 1, 2, 2, 3, 0,  // front face
    4, 6, 5, 6, 7, 4,  // back face
    0, 4, 5, 5, 1, 0,  // bottom face
    2, 6, 7, 7, 3, 4,  // top face
    0, 3, 7, 7, 4, 0,  // left face
    1, 5, 6, 6, 2, 1   // right face
}};"""
    else:
        # Sphere placeholder
        verts = """// Sphere vertices - use generateSphereVertices() below
static std::vector<Vertex> sphereVertices;
static std::vector<uint32_t> sphereIndices;"""

    return f"""#include <array>
#include <cstdint>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>
#include <vk_mem_allocate.h>

namespace vulkanmind {{

struct Vertex {{
    float x, y, z;
    float r, g, b;
}};

{verts}

class GeometryApp {{
public:
    void run() {{
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }}
private:
    void initWindow() {{
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan {shape.title()}", nullptr, nullptr);
    }}

    void initVulkan() {{
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createVertexBuffer();
        createCommandBuffers();
        createSyncObjects();
    }}

    void mainLoop() {{
        while (!glfwWindowShouldClose(window_)) {{
            glfwPollEvents();
            drawFrame();
        }}
        device_.waitIdle();
    }}

    void cleanup() {{
        cleanupSyncObjects();
        device_.destroyBuffer(vertexBuffer_);
        device_.freeMemory(vertexBufferMemory_);
        for (auto framebuffer : swapchainFramebuffers_) {{
            device_.destroyFramebuffer(framebuffer);
        }}
        device_.destroyPipeline(graphicsPipeline_);
        device_.destroyPipelineLayout(pipelineLayout_);
        device_.destroyRenderPass(renderPass_);
        for (auto imageView : swapchainImageViews_) {{
            device_.destroyImageView(imageView);
        }}
        device_.destroySwapchainKHR(swapchain_);
        device_.destroyDevice();
        instance_.destroySurfaceKHR(surface_);
        instance_.destroy();
        glfwDestroyWindow(window_);
        glfwTerminate();
    }}

    void drawFrame() {{
        uint32_t imageIndex;
        vk::Result result = device_.acquireNextImageKHR(swapchain_, UINT64_MAX,
            imageAvailableSemaphores_[currentFrame_], nullptr, &imageIndex);

        if (result == vk::Result::eErrorOutOfDateKHR) {{
            recreateSwapchain();
            return;
        }}

        vk::SubmitInfo submitInfo{{}};
        vk::Semaphore waitSemaphores[] = {{imageAvailableSemaphores_[currentFrame_]}};
        vk::PipelineStageFlags waitStages[] = {{vk::PipelineStageFlagBits::eColorAttachmentOutput}};
        submitInfo.setWaitSemaphores(waitSemaphores)
                 .setWaitDstStageMask(waitStages)
                 .setCommandBuffers(commandBuffers_[imageIndex])
                 .setSignalSemaphores(renderFinishedSemaphores_[currentFrame_]);

        device_.resetFences(1, &inFlightFences_[currentFrame_]);
        graphicsQueue_.submit(submitInfo, inFlightFences_[currentFrame_]);

        vk::PresentInfoKHR presentInfo{{}};
        presentInfo.setWaitSemaphores(renderFinishedSemaphores_[currentFrame_])
                  .setSwapchains(swapchain_)
                  .setImageIndices(imageIndex);

        graphicsQueue_.presentKHR(presentInfo);
    }}

    void createGraphicsPipeline() {{
        // TODO: Implement shader loading and pipeline creation
    }}

    static constexpr int WIDTH = 800;
    static constexpr int HEIGHT = 600;
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    GLFWwindow* window_ = nullptr;
    vk::Instance instance_;
    vk::SurfaceKHR surface_;
    vk::PhysicalDevice physicalDevice_;
    vk::Device device_;
    vk::Queue graphicsQueue_;
    vk::SwapchainKHR swapchain_;
    std::vector<vk::Image> swapchainImages_;
    vk::Extent2D swapchainExtent_;
    std::vector<vk::ImageView> swapchainImageViews_;
    vk::RenderPass renderPass_;
    vk::PipelineLayout pipelineLayout_;
    vk::Pipeline graphicsPipeline_;
    std::vector<vk::Framebuffer> swapchainFramebuffers_;
    vk::CommandPool commandPool_;
    vk::Buffer vertexBuffer_;
    vk::DeviceMemory vertexBufferMemory_;
    std::vector<vk::CommandBuffer> commandBuffers_;
    std::vector<vk::Semaphore> imageAvailableSemaphores_;
    std::vector<vk::Semaphore> renderFinishedSemaphores_;
    std::vector<vk::Fence> inFlightFences_;
    int currentFrame_ = 0;
}};

int main() {{
    vulkanmind::GeometryApp app;
    app.run();
    return 0;
}}
"""


def generate_llm_code(user_request: str, context: PlatformContext) -> str:
    """Generate code via LLM analysis for arbitrary requests."""
    # Use platform-specific context in the prompt
    vendor = context.target.gpu_vendor
    quirks = ", ".join(f"{k}={v}" for k, v in context.target.quirk_profile.items())
    return f"""// Vulkan code generator output for: "{user_request}"
// Target platform: {vendor}, Quirks: {quirks}

#include <array>
#include <cstdint>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>
#include <vk_mem_allocate.h>

namespace vulkanmind {{

// Request: {user_request}
// This is a scaffold - extend for your specific geometry

struct Vertex {{
    float x, y, z;
    float r, g, b;
}};

class GeneratedApp {{
public:
    void run() {{
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }}
private:
    void initWindow() {{
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(800, 600, "Generated", nullptr, nullptr);
    }}

    void initVulkan() {{
        // TODO: Implement initialization
    }}

    void mainLoop() {{
        while (!glfwWindowShouldClose(window_)) {{
            glfwPollEvents();
        }}
    }}

    void cleanup() {{
        glfwDestroyWindow(window_);
        glfwTerminate();
    }}

    GLFWwindow* window_ = nullptr;
    vk::Instance instance_;
    vk::Device device_;
}};

int main() {{
    vulkanmind::GeneratedApp app;
    app.run();
    return 0;
}}
"""


def generate_code_for_request(user_request: str, context: PlatformContext) -> str:
    """Generate Vulkan code based on the user request."""
    lowered = user_request.lower()

    # Geometry-specific generators
    if "triangle" in lowered or "simple render" in lowered:
        return generate_triangle_code(context)

    if "circle" in lowered or "texture" in lowered:
        return generate_textured_shape_code(context, "circle")

    if "cube" in lowered or "prism" in lowered or "rectangular" in lowered:
        return generate_geometry_code(context, "cube")

    if "sphere" in lowered or "ico" in lowered:
        return generate_geometry_code(context, "sphere")

    if "swapchain" in lowered or "image" in lowered:
        return generate_swapchain_code(context)

    if "compute" in lowered or "compute shader" in lowered:
        return generate_vulkan_hpp_boilerplate(context)

    # Fallback: generate code via LLM for arbitrary requests
    return generate_llm_code(user_request, context)