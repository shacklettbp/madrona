#include "viewer_renderer.hpp"

#include <madrona/render/shader_compiler.hpp>

#include "backends/imgui_impl_vulkan.h"
#include "backends/imgui_impl_glfw.h"

#include "../render/asset_utils.hpp"

#include "vk/descriptors.hpp"

#include <filesystem>

#ifdef MADRONA_MACOS
#include <dlfcn.h>
#endif

#ifdef MADRONA_CUDA_SUPPORT
#include "vk/cuda_interop.hpp"
#endif

#include "shader.hpp"

using namespace std;

using namespace madrona::render;
using namespace madrona::render::vk;

namespace madrona::viz {

using Vertex = shader::Vertex;
using PackedVertex = shader::PackedVertex;
using MeshData = shader::MeshData;
using MaterialData = shader::MaterialData;
using ObjectData = shader::ObjectData;
using DrawPushConst = shader::DrawPushConst;
using CullPushConst = shader::CullPushConst;
using DrawCmd = shader::DrawCmd;
using DrawMaterialData = shader::DrawMaterialData;
using PackedInstanceData = shader::PackedInstanceData;
using PackedViewData = shader::PackedViewData;
using ShadowViewData = shader::ShadowViewData;
using DirectionalLight = shader::DirectionalLight;

namespace InternalConfig {

inline constexpr uint32_t numFrames = 2;
inline constexpr uint32_t initMaxTransforms = 100000;
inline constexpr uint32_t initMaxMatIndices = 100000;
inline constexpr uint32_t shadowMapSize = 4096;
inline constexpr uint32_t maxLights = 10;

}

struct ImGUIVkLookupData {
    PFN_vkGetDeviceProcAddr getDevAddr;
    VkDevice dev;
    PFN_vkGetInstanceProcAddr getInstAddr;
    VkInstance inst;
};

PFN_vkGetInstanceProcAddr PresentationState::init()
{
#ifdef MADRONA_MACOS
    auto inst_addr = (PFN_vkGetInstanceProcAddr)dlsym(
        RTLD_DEFAULT, "vkGetInstanceProcAddr");
    glfwInitVulkanLoader(inst_addr);
#endif

    if (!glfwInit()) {
        FATAL("Failed to initialize GLFW");
    }

    return (PFN_vkGetInstanceProcAddr)glfwGetInstanceProcAddress(
        VK_NULL_HANDLE, "vkGetInstanceProcAddr");
}

vector<const char *> PresentationState::getInstanceExtensions()
{
    uint32_t count;
    const char **names = glfwGetRequiredInstanceExtensions(&count);

    vector<const char *> exts(count);
    memcpy(exts.data(), names, count * sizeof(const char *));

    return exts;
}

static GLFWwindow *makeGLFWwindow(uint32_t width, uint32_t height)
{
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);

#ifdef MADRONA_MACOS
    width = utils::divideRoundUp(width, 2_u32);
    height = utils::divideRoundUp(height, 2_u32);
#endif

#if 0
    auto monitor = glfwGetPrimaryMonitor();
    auto mode = glfwGetVideoMode(monitor);

    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
    return glfwCreateWindow(mode->width, mode->height, "Madrona", monitor, nullptr);
#endif

    return glfwCreateWindow(width, height, "Madrona", nullptr, nullptr);
}

VkSurfaceKHR getWindowSurface(const Backend &backend, GLFWwindow *window)
{
    VkSurfaceKHR surface;
    REQ_VK(glfwCreateWindowSurface(backend.hdl, window, nullptr, &surface));

    return surface;
}

static VkSurfaceFormatKHR selectSwapchainFormat(const Backend &backend,
                                                VkPhysicalDevice phy,
                                                VkSurfaceKHR surface)
{
    uint32_t num_formats;
    REQ_VK(backend.dt.getPhysicalDeviceSurfaceFormatsKHR(
            phy, surface, &num_formats, nullptr));

    HeapArray<VkSurfaceFormatKHR> formats(num_formats);
    REQ_VK(backend.dt.getPhysicalDeviceSurfaceFormatsKHR(
            phy, surface, &num_formats, formats.data()));

    if (num_formats == 0) {
        FATAL("Zero swapchain formats");
    }

    // FIXME
    for (VkSurfaceFormatKHR format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }

    return formats[0];
}

static VkPresentModeKHR selectSwapchainMode(const Backend &backend,
                                            VkPhysicalDevice phy,
                                            VkSurfaceKHR surface,
                                            bool need_immediate)
{
    uint32_t num_modes;
    REQ_VK(backend.dt.getPhysicalDeviceSurfacePresentModesKHR(
            phy, surface, &num_modes, nullptr));

    HeapArray<VkPresentModeKHR> modes(num_modes);
    REQ_VK(backend.dt.getPhysicalDeviceSurfacePresentModesKHR(
            phy, surface, &num_modes, modes.data()));

    for (VkPresentModeKHR mode : modes) {
        if (need_immediate && mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
            return mode;
        } else if (!need_immediate && mode == VK_PRESENT_MODE_FIFO_KHR) {
            return mode;
        }
    }

    if (!need_immediate) {
        return modes[0];
    } else {
        FATAL("Could not find immediate swapchain");
    }
}

static Swapchain makeSwapchain(const Backend &backend,
                               const Device &dev,
                               const Window &window,
                               uint32_t num_frames_inflight,
                               bool need_immediate)
{
    VkSurfaceFormatKHR format =
        selectSwapchainFormat(backend, dev.phy, window.surface);
    VkPresentModeKHR mode =
        selectSwapchainMode(backend, dev.phy, window.surface, need_immediate);

    VkSurfaceCapabilitiesKHR caps;
    REQ_VK(backend.dt.getPhysicalDeviceSurfaceCapabilitiesKHR(
            dev.phy, window.surface, &caps));

    VkExtent2D swapchain_size = caps.currentExtent;
    if (swapchain_size.width == UINT32_MAX &&
        swapchain_size.height == UINT32_MAX) {
        glfwGetWindowSize(window.platformWindow, (int *)&swapchain_size.width,
                          (int *)&swapchain_size.height);

        swapchain_size.width = max(caps.minImageExtent.width,
                                   min(caps.maxImageExtent.width,
                                       swapchain_size.width));

        swapchain_size.height = max(caps.minImageExtent.height,
                                    min(caps.maxImageExtent.height,
                                        swapchain_size.height));
    }

    uint32_t num_requested_images =
        max(caps.minImageCount + 1, num_frames_inflight);
    if (caps.maxImageCount != 0 && num_requested_images > caps.maxImageCount) {
        num_requested_images = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swapchain_info;
    swapchain_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchain_info.pNext = nullptr;
    swapchain_info.flags = 0;
    swapchain_info.surface = window.surface;
    swapchain_info.minImageCount = num_requested_images;
    swapchain_info.imageFormat = format.format;
    swapchain_info.imageColorSpace = format.colorSpace;
    swapchain_info.imageExtent = swapchain_size;
    swapchain_info.imageArrayLayers = 1;
    swapchain_info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swapchain_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_info.queueFamilyIndexCount = 0;
    swapchain_info.pQueueFamilyIndices = nullptr;
    swapchain_info.preTransform = caps.currentTransform;
    swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchain_info.presentMode = mode;
    swapchain_info.clipped = VK_TRUE;
    swapchain_info.oldSwapchain = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain;
    REQ_VK(dev.dt.createSwapchainKHR(dev.hdl, &swapchain_info, nullptr,
                                     &swapchain));

    return Swapchain {
        swapchain,
        swapchain_size.width,
        swapchain_size.height,
    };
}

static HeapArray<VkImage> getSwapchainImages(const Device &dev,
                                            VkSwapchainKHR swapchain)
{
    uint32_t num_images;
    REQ_VK(dev.dt.getSwapchainImagesKHR(dev.hdl, swapchain, &num_images,
                                        nullptr));

    HeapArray<VkImage> swapchain_images(num_images);
    REQ_VK(dev.dt.getSwapchainImagesKHR(dev.hdl, swapchain, &num_images,
                                        swapchain_images.data()));

    return swapchain_images;
}

static Window makeWindow(const Backend &backend,
                         uint32_t window_width,
                         uint32_t window_height)
{
    GLFWwindow *glfw_window =
        makeGLFWwindow(window_width, window_height);

    VkSurfaceKHR surface = getWindowSurface(backend, glfw_window);

    return Window {
        .platformWindow = glfw_window,
        .surface = surface,
        .width = window_width,
        .height = window_height,
    };
}

PresentationState::PresentationState(const Backend &backend,
                                     const Device &dev,
                                     const Window &window,
                                     uint32_t num_frames_inflight,
                                     bool need_immediate)
    : swapchain_(makeSwapchain(backend, dev, window,
                               num_frames_inflight,
                               need_immediate)),
      swapchain_imgs_(getSwapchainImages(dev, swapchain_.hdl))
{
}

void PresentationState::destroy(const Device &dev)
{
    dev.dt.destroySwapchainKHR(dev.hdl, swapchain_.hdl, nullptr);
}

void PresentationState::forceTransition(const Device &dev,
    const QueueState &present_queue, uint32_t qf_idx)
{
    VkCommandPool tmp_pool = makeCmdPool(dev, qf_idx);
    VkCommandBuffer cmd = makeCmdBuffer(dev, tmp_pool);

    vector<VkImageMemoryBarrier> barriers;
    barriers.reserve(swapchain_imgs_.size());

    for (int i = 0; i < (int)swapchain_imgs_.size(); i++) {
        barriers.push_back({
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            0,
            0,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            swapchain_imgs_[i],
            { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        });
    }

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    REQ_VK(dev.dt.beginCommandBuffer(cmd, &begin_info));

    dev.dt.cmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              0, 0, nullptr, 0, nullptr,
                              barriers.size(), barriers.data());

    REQ_VK(dev.dt.endCommandBuffer(cmd));

    VkSubmitInfo render_submit {};
    render_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    render_submit.waitSemaphoreCount = 0;
    render_submit.pWaitSemaphores = nullptr;
    render_submit.pWaitDstStageMask = nullptr;
    render_submit.commandBufferCount = 1;
    render_submit.pCommandBuffers = &cmd;

    VkFence fence = makeFence(dev);

    present_queue.submit(dev, 1, &render_submit, fence);

    waitForFenceInfinitely(dev, fence);

    // FIXME, get an initialization pool / fence for stuff like this
    dev.dt.destroyFence(dev.hdl, fence, nullptr);
    dev.dt.destroyCommandPool(dev.hdl, tmp_pool, nullptr);
}

uint32_t PresentationState::acquireNext(const Device &dev,
                                        VkSemaphore signal_sema)
{
    uint32_t swapchain_idx;
    REQ_VK(dev.dt.acquireNextImageKHR(dev.hdl, swapchain_.hdl,
                                      0, signal_sema,
                                      VK_NULL_HANDLE,
                                      &swapchain_idx));

    return swapchain_idx;
}

VkImage PresentationState::getImage(uint32_t idx) const
{
    return swapchain_imgs_[idx];
}

uint32_t PresentationState::numSwapchainImages() const
{
    return swapchain_imgs_.size();
}

void PresentationState::present(const Device &dev, uint32_t swapchain_idx,
                                const QueueState &present_queue,
                                uint32_t num_wait_semas,
                                const VkSemaphore *wait_semas)
{
    VkPresentInfoKHR present_info;
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.pNext = nullptr;
    present_info.waitSemaphoreCount = num_wait_semas;
    present_info.pWaitSemaphores = wait_semas;

    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain_.hdl;
    present_info.pImageIndices = &swapchain_idx;
    present_info.pResults = nullptr;

    present_queue.presentSubmit(dev, &present_info);
}

static VkQueue makeGFXQueue(const Device &dev, uint32_t idx)
{
    if (idx >= dev.numGraphicsQueues) {
        FATAL("Not enough graphics queues");
    }

    return makeQueue(dev, dev.gfxQF, idx);
}

static VkQueue makeComputeQueue(const Device &dev, uint32_t idx)
{
    if (idx >= dev.numComputeQueues) {
        FATAL("Not enough compute queues");
    }

    return makeQueue(dev, dev.computeQF, idx);
}

static VkQueue makeTransferQueue(const Device &dev, uint32_t idx)
{
    if (idx >= dev.numTransferQueues) {
        FATAL("Not enough transfer queues");
    }

    return makeQueue(dev, dev.transferQF, idx);
}

static VkRenderPass makeRenderPass(const Device &dev,
                                   VkFormat color_fmt,
                                   VkFormat depth_fmt)
{
    vector<VkAttachmentDescription> attachment_descs;
    vector<VkAttachmentReference> attachment_refs;

    attachment_descs.push_back(
        {0, color_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    attachment_refs.push_back(
        {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    
    attachment_descs.push_back(
        {0, depth_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

    attachment_refs.push_back(
        {static_cast<uint32_t>(attachment_refs.size()),
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

    VkSubpassDescription subpass_desc {};
    subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_desc.colorAttachmentCount =
        static_cast<uint32_t>(attachment_refs.size() - 1);
    subpass_desc.pColorAttachments = &attachment_refs[0];
    subpass_desc.pDepthStencilAttachment = &attachment_refs.back();

    VkRenderPassCreateInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.flags = 0;
    render_pass_info.attachmentCount =
        static_cast<uint32_t>(attachment_descs.size());
    render_pass_info.pAttachments = attachment_descs.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass_desc;

    render_pass_info.dependencyCount = 0;
    render_pass_info.pDependencies = nullptr;

    VkRenderPass render_pass;
    REQ_VK(dev.dt.createRenderPass(dev.hdl, &render_pass_info, nullptr,
                                   &render_pass));

    return render_pass;
}

static VkRenderPass makeShadowRenderPass(const Device &dev,
                                         VkFormat depth_fmt)
{
    vector<VkAttachmentDescription> attachment_descs;
    vector<VkAttachmentReference> attachment_refs;

    attachment_descs.push_back(
        {0, depth_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

    attachment_refs.push_back(
        {static_cast<uint32_t>(attachment_refs.size()),
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

    VkSubpassDescription subpass_desc {};
    subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_desc.colorAttachmentCount = 0;
    subpass_desc.pDepthStencilAttachment = &attachment_refs.back();

    VkRenderPassCreateInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.flags = 0;
    render_pass_info.attachmentCount =
        static_cast<uint32_t>(attachment_descs.size());
    render_pass_info.pAttachments = attachment_descs.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass_desc;

    render_pass_info.dependencyCount = 0;
    render_pass_info.pDependencies = nullptr;

    VkRenderPass render_pass;
    REQ_VK(dev.dt.createRenderPass(dev.hdl, &render_pass_info, nullptr,
                                   &render_pass));

    return render_pass;
}

static PipelineShaders makeDrawShaders(
    const Device &dev, VkSampler repeat_sampler, VkSampler clamp_sampler)
{
    (void)repeat_sampler;
    (void)clamp_sampler;

    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(VIEWER_DATA_DIR)) /
        "shaders";

    auto shader_path = (shader_dir / "viewer_draw.hlsl");

    ShaderCompiler compiler;
    SPIRVShader vert_spirv = compiler.compileHLSLFileToSPV(
        shader_path.c_str(), {}, {},
        { "vert", ShaderStage::Vertex });

    SPIRVShader frag_spirv = compiler.compileHLSLFileToSPV(
        shader_path.c_str(), {}, {},
        { "frag", ShaderStage::Fragment });

#if 0
            {0, 2, repeat_sampler, 1, 0},
            {0, 3, clamp_sampler, 1, 0},
            {1, 1, VK_NULL_HANDLE,
                VulkanConfig::max_materials *
                    VulkanConfig::textures_per_material,
             VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},
#endif

    std::array<SPIRVShader, 2> shaders {
        std::move(vert_spirv),
        std::move(frag_spirv),
    };

    StackAlloc tmp_alloc;
    return PipelineShaders(dev, tmp_alloc, shaders, {});
}

static PipelineShaders makeShadowDrawShaders(
    const Device &dev, VkSampler repeat_sampler, VkSampler clamp_sampler)
{
    (void)repeat_sampler;
    (void)clamp_sampler;

    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(VIEWER_DATA_DIR)) /
        "shaders";

    auto shader_path = (shader_dir / "viewer_shadow_draw.hlsl");

    ShaderCompiler compiler;
    SPIRVShader vert_spirv = compiler.compileHLSLFileToSPV(
        shader_path.c_str(), {}, {},
        { "vert", ShaderStage::Vertex });

    SPIRVShader frag_spirv = compiler.compileHLSLFileToSPV(
        shader_path.c_str(), {}, {},
        { "frag", ShaderStage::Fragment });

#if 0
            {0, 2, repeat_sampler, 1, 0},
            {0, 3, clamp_sampler, 1, 0},
            {1, 1, VK_NULL_HANDLE,
                VulkanConfig::max_materials *
                    VulkanConfig::textures_per_material,
             VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},
#endif

    std::array<SPIRVShader, 2> shaders {
        std::move(vert_spirv),
        std::move(frag_spirv),
    };

    StackAlloc tmp_alloc;
    return PipelineShaders(dev, tmp_alloc, shaders, {});
}

static PipelineShaders makeCullShader(const Device &dev)
{
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(VIEWER_DATA_DIR)) /
        "shaders";

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / "viewer_cull.hlsl").c_str(), {}, {},
        { "instanceCull", ShaderStage::Compute });

    StackAlloc tmp_alloc;
    return PipelineShaders(dev, tmp_alloc,
                           Span<const SPIRVShader>(&spirv, 1), {});
}

static VkPipelineCache getPipelineCache(const Device &dev)
{
    // Pipeline cache (unsaved)
    VkPipelineCacheCreateInfo pcache_info {};
    pcache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VkPipelineCache pipeline_cache;
    REQ_VK(dev.dt.createPipelineCache(dev.hdl, &pcache_info, nullptr,
                                      &pipeline_cache));

    return pipeline_cache;
}

static Pipeline<1> makeDrawPipeline(const Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    VkRenderPass render_pass,
                                    VkSampler repeat_sampler,
                                    VkSampler clamp_sampler,
                                    uint32_t num_frames)
{
    auto shaders =
        makeDrawShaders(dev, repeat_sampler, clamp_sampler);

    // Disable auto vertex assembly
    VkPipelineVertexInputStateCreateInfo vert_info;
    vert_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vert_info.pNext = nullptr;
    vert_info.flags = 0;
    vert_info.vertexBindingDescriptionCount = 0;
    vert_info.pVertexBindingDescriptions = nullptr;
    vert_info.vertexAttributeDescriptionCount = 0;
    vert_info.pVertexAttributeDescriptions = nullptr;

    // Assembly (standard tri indices)
    VkPipelineInputAssemblyStateCreateInfo input_assembly_info {};
    input_assembly_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly_info.primitiveRestartEnable = VK_FALSE;

    // Viewport (fully dynamic)
    VkPipelineViewportStateCreateInfo viewport_info {};
    viewport_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_info.viewportCount = 1;
    viewport_info.pViewports = nullptr;
    viewport_info.scissorCount = 1;
    viewport_info.pScissors = nullptr;

    // Multisample
    VkPipelineMultisampleStateCreateInfo multisample_info {};
    multisample_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample_info.sampleShadingEnable = VK_FALSE;
    multisample_info.alphaToCoverageEnable = VK_FALSE;
    multisample_info.alphaToOneEnable = VK_FALSE;

    // Rasterization
    VkPipelineRasterizationStateCreateInfo raster_info {};
    raster_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster_info.depthClampEnable = VK_FALSE;
    raster_info.rasterizerDiscardEnable = VK_FALSE;
    raster_info.polygonMode = VK_POLYGON_MODE_FILL;
    raster_info.cullMode = VK_CULL_MODE_BACK_BIT;
    raster_info.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster_info.depthBiasEnable = VK_FALSE;
    raster_info.lineWidth = 1.0f;

    // Depth/Stencil
    VkPipelineDepthStencilStateCreateInfo depth_info {};
    depth_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_info.depthTestEnable = VK_TRUE;
    depth_info.depthWriteEnable = VK_TRUE;
    depth_info.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
    depth_info.depthBoundsTestEnable = VK_FALSE;
    depth_info.stencilTestEnable = VK_FALSE;
    depth_info.back.compareOp = VK_COMPARE_OP_ALWAYS;

    // Blend
    VkPipelineColorBlendAttachmentState blend_attach {};
    blend_attach.blendEnable = VK_FALSE;
    blend_attach.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    array<VkPipelineColorBlendAttachmentState, 1> blend_attachments {{
        blend_attach,
    }};

    VkPipelineColorBlendStateCreateInfo blend_info {};
    blend_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend_info.logicOpEnable = VK_FALSE;
    blend_info.attachmentCount =
        static_cast<uint32_t>(blend_attachments.size());
    blend_info.pAttachments = blend_attachments.data();

    // Dynamic
    array dyn_enable {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dyn_info {};
    dyn_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn_info.dynamicStateCount = dyn_enable.size();
    dyn_info.pDynamicStates = dyn_enable.data();

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(DrawPushConst),
    };

    // Layout configuration

    array<VkDescriptorSetLayout, 2> draw_desc_layouts {{
        shaders.getLayout(0),
        shaders.getLayout(1),
    }};

    VkPipelineLayoutCreateInfo gfx_layout_info;
    gfx_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    gfx_layout_info.pNext = nullptr;
    gfx_layout_info.flags = 0;
    gfx_layout_info.setLayoutCount =
        static_cast<uint32_t>(draw_desc_layouts.size());
    gfx_layout_info.pSetLayouts = draw_desc_layouts.data();
    gfx_layout_info.pushConstantRangeCount = 1;
    gfx_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout draw_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &gfx_layout_info, nullptr,
                                       &draw_layout));

    array<VkPipelineShaderStageCreateInfo, 2> gfx_stages {{
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_VERTEX_BIT,
            shaders.getShader(0),
            "vert",
            nullptr,
        },
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            shaders.getShader(1),
            "frag",
            nullptr,
        },
    }};

    VkGraphicsPipelineCreateInfo gfx_info;
    gfx_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gfx_info.pNext = nullptr;
    gfx_info.flags = 0;
    gfx_info.stageCount = gfx_stages.size();
    gfx_info.pStages = gfx_stages.data();
    gfx_info.pVertexInputState = &vert_info;
    gfx_info.pInputAssemblyState = &input_assembly_info;
    gfx_info.pTessellationState = nullptr;
    gfx_info.pViewportState = &viewport_info;
    gfx_info.pRasterizationState = &raster_info;
    gfx_info.pMultisampleState = &multisample_info;
    gfx_info.pDepthStencilState = &depth_info;
    gfx_info.pColorBlendState = &blend_info;
    gfx_info.pDynamicState = &dyn_info;
    gfx_info.layout = draw_layout;
    gfx_info.renderPass = render_pass;
    gfx_info.subpass = 0;
    gfx_info.basePipelineHandle = VK_NULL_HANDLE;
    gfx_info.basePipelineIndex = -1;

    VkPipeline draw_pipeline;
    REQ_VK(dev.dt.createGraphicsPipelines(dev.hdl, pipeline_cache, 1,
                                          &gfx_info, nullptr, &draw_pipeline));

    FixedDescriptorPool desc_pool(dev, shaders, 0, num_frames);

    return {
        std::move(shaders),
        draw_layout,
        { draw_pipeline },
        std::move(desc_pool),
    };
}

static Pipeline<1> makeShadowDrawPipeline(const Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    VkRenderPass render_pass,
                                    VkSampler repeat_sampler,
                                    VkSampler clamp_sampler,
                                    uint32_t num_frames)
{
    auto shaders =
        makeShadowDrawShaders(dev, repeat_sampler, clamp_sampler);

    // Disable auto vertex assembly
    VkPipelineVertexInputStateCreateInfo vert_info;
    vert_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vert_info.pNext = nullptr;
    vert_info.flags = 0;
    vert_info.vertexBindingDescriptionCount = 0;
    vert_info.pVertexBindingDescriptions = nullptr;
    vert_info.vertexAttributeDescriptionCount = 0;
    vert_info.pVertexAttributeDescriptions = nullptr;

    // Assembly (standard tri indices)
    VkPipelineInputAssemblyStateCreateInfo input_assembly_info {};
    input_assembly_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly_info.primitiveRestartEnable = VK_FALSE;

    // Viewport (fully dynamic)
    VkPipelineViewportStateCreateInfo viewport_info {};
    viewport_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_info.viewportCount = 1;
    viewport_info.pViewports = nullptr;
    viewport_info.scissorCount = 1;
    viewport_info.pScissors = nullptr;

    // Multisample
    VkPipelineMultisampleStateCreateInfo multisample_info {};
    multisample_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample_info.sampleShadingEnable = VK_FALSE;
    multisample_info.alphaToCoverageEnable = VK_FALSE;
    multisample_info.alphaToOneEnable = VK_FALSE;

    // Rasterization
    VkPipelineRasterizationStateCreateInfo raster_info {};
    raster_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster_info.depthClampEnable = VK_FALSE;
    raster_info.rasterizerDiscardEnable = VK_FALSE;
    raster_info.polygonMode = VK_POLYGON_MODE_FILL;
    raster_info.cullMode = VK_CULL_MODE_BACK_BIT;
    raster_info.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster_info.depthBiasEnable = VK_FALSE;
    raster_info.lineWidth = 1.0f;

    // Depth/Stencil
    VkPipelineDepthStencilStateCreateInfo depth_info {};
    depth_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_info.depthTestEnable = VK_TRUE;
    depth_info.depthWriteEnable = VK_TRUE;
    depth_info.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
    depth_info.depthBoundsTestEnable = VK_FALSE;
    depth_info.stencilTestEnable = VK_FALSE;
    depth_info.back.compareOp = VK_COMPARE_OP_ALWAYS;

    VkPipelineColorBlendStateCreateInfo blend_info {};
    blend_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend_info.logicOpEnable = VK_FALSE;
    blend_info.attachmentCount = 0;

    // Dynamic
    array dyn_enable {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dyn_info {};
    dyn_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn_info.dynamicStateCount = dyn_enable.size();
    dyn_info.pDynamicStates = dyn_enable.data();

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(DrawPushConst),
    };

    // Layout configuration

    array<VkDescriptorSetLayout, 2> draw_desc_layouts {{
        shaders.getLayout(0),
        shaders.getLayout(1),
    }};

    VkPipelineLayoutCreateInfo gfx_layout_info;
    gfx_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    gfx_layout_info.pNext = nullptr;
    gfx_layout_info.flags = 0;
    gfx_layout_info.setLayoutCount =
        static_cast<uint32_t>(draw_desc_layouts.size());
    gfx_layout_info.pSetLayouts = draw_desc_layouts.data();
    gfx_layout_info.pushConstantRangeCount = 1;
    gfx_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout draw_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &gfx_layout_info, nullptr,
                                       &draw_layout));

    array<VkPipelineShaderStageCreateInfo, 2> gfx_stages {{
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_VERTEX_BIT,
            shaders.getShader(0),
            "vert",
            nullptr,
        },
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            shaders.getShader(1),
            "frag",
            nullptr,
        },
    }};

    VkGraphicsPipelineCreateInfo gfx_info;
    gfx_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gfx_info.pNext = nullptr;
    gfx_info.flags = 0;
    gfx_info.stageCount = gfx_stages.size();
    gfx_info.pStages = gfx_stages.data();
    gfx_info.pVertexInputState = &vert_info;
    gfx_info.pInputAssemblyState = &input_assembly_info;
    gfx_info.pTessellationState = nullptr;
    gfx_info.pViewportState = &viewport_info;
    gfx_info.pRasterizationState = &raster_info;
    gfx_info.pMultisampleState = &multisample_info;
    gfx_info.pDepthStencilState = &depth_info;
    gfx_info.pColorBlendState = &blend_info;
    gfx_info.pDynamicState = &dyn_info;
    gfx_info.layout = draw_layout;
    gfx_info.renderPass = render_pass;
    gfx_info.subpass = 0;
    gfx_info.basePipelineHandle = VK_NULL_HANDLE;
    gfx_info.basePipelineIndex = -1;

    VkPipeline draw_pipeline;
    REQ_VK(dev.dt.createGraphicsPipelines(dev.hdl, pipeline_cache, 1,
                                          &gfx_info, nullptr, &draw_pipeline));

    FixedDescriptorPool desc_pool(dev, shaders, 0, num_frames);

    return {
        std::move(shaders),
        draw_layout,
        { draw_pipeline },
        std::move(desc_pool),
    };
}

static Pipeline<1> makeCullPipeline(const Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    CountT num_frames)
{
    PipelineShaders shader = makeCullShader(dev);

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(CullPushConst),
    };

    // Layout configuration
    std::array desc_layouts {
        shader.getLayout(0),
        shader.getLayout(1),
    };

    VkPipelineLayoutCreateInfo cull_layout_info;
    cull_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    cull_layout_info.pNext = nullptr;
    cull_layout_info.flags = 0;
    cull_layout_info.setLayoutCount =
        static_cast<uint32_t>(desc_layouts.size());
    cull_layout_info.pSetLayouts = desc_layouts.data();
    cull_layout_info.pushConstantRangeCount = 1;
    cull_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout cull_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &cull_layout_info, nullptr,
                                       &cull_layout));

    std::array<VkComputePipelineCreateInfo, 1> compute_infos;
#if 0
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_size;
    subgroup_size.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
    subgroup_size.pNext = nullptr;
    subgroup_size.requiredSubgroupSize = 32;
#endif

    compute_infos[0].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_infos[0].pNext = nullptr;
    compute_infos[0].flags = 0;
    compute_infos[0].stage = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, //&subgroup_size,
        VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
        VK_SHADER_STAGE_COMPUTE_BIT,
        shader.getShader(0),
        "instanceCull",
        nullptr,
    };
    compute_infos[0].layout = cull_layout;
    compute_infos[0].basePipelineHandle = VK_NULL_HANDLE;
    compute_infos[0].basePipelineIndex = -1;

    std::array<VkPipeline, compute_infos.size()> pipelines;
    REQ_VK(dev.dt.createComputePipelines(dev.hdl, pipeline_cache,
                                         compute_infos.size(),
                                         compute_infos.data(), nullptr,
                                         pipelines.data()));

    FixedDescriptorPool desc_pool(dev, shader, 0, num_frames);

    return Pipeline<1> {
        std::move(shader),
        cull_layout,
        pipelines,
        std::move(desc_pool),
    };
}

static Backend initializeBackend()
{
    auto get_inst_addr = PresentationState::init();

    bool enable_validation;
    char *validate_env = getenv("MADRONA_RENDER_VALIDATE");
    if (!validate_env || validate_env[0] == '0') {
        enable_validation = false;
    } else {
        enable_validation = true;
    }

    return Backend((void (*)())get_inst_addr, enable_validation, false,
                   PresentationState::getInstanceExtensions());
}

static Framebuffer makeFramebuffer(const Device &dev,
                                   MemoryAllocator &alloc,
                                   uint32_t fb_width,
                                   uint32_t fb_height,
                                   VkRenderPass render_pass)
{
    auto color = alloc.makeColorAttachment(fb_width, fb_height);
    auto depth = alloc.makeDepthAttachment(fb_width, fb_height);

    VkImageViewCreateInfo view_info {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkImageSubresourceRange &view_info_sr = view_info.subresourceRange;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info_sr.baseMipLevel = 0;
    view_info_sr.levelCount = 1;
    view_info_sr.baseArrayLayer = 0;
    view_info_sr.layerCount = 1;

    view_info.image = color.image;
    view_info.format = alloc.getColorAttachmentFormat();

    VkImageView color_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &color_view));

    view_info.image = depth.image;
    view_info.format = alloc.getDepthAttachmentFormat();
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    VkImageView depth_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &depth_view));

    array attachment_views {
        color_view,
        depth_view,
    };

    VkFramebufferCreateInfo fb_info;
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.pNext = nullptr;
    fb_info.flags = 0;
    fb_info.renderPass = render_pass;
    fb_info.attachmentCount = static_cast<uint32_t>(attachment_views.size());
    fb_info.pAttachments = attachment_views.data();
    fb_info.width = fb_width;
    fb_info.height = fb_height;
    fb_info.layers = 1;

    VkFramebuffer hdl;
    REQ_VK(dev.dt.createFramebuffer(dev.hdl, &fb_info, nullptr, &hdl));

    return Framebuffer {
        std::move(color),
        std::move(depth),
        color_view,
        depth_view,
        hdl,
    };
}

static ShadowFramebuffer makeShadowFramebuffer(const Device &dev,
                                   MemoryAllocator &alloc,
                                   uint32_t fb_width,
                                   uint32_t fb_height,
                                   VkRenderPass render_pass)
{
    auto depth = alloc.makeDepthAttachment(fb_width, fb_height);

    VkImageViewCreateInfo view_info {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkImageSubresourceRange &view_info_sr = view_info.subresourceRange;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    view_info_sr.baseMipLevel = 0;
    view_info_sr.levelCount = 1;
    view_info_sr.baseArrayLayer = 0;
    view_info_sr.layerCount = 1;

    view_info.image = depth.image;
    view_info.format = alloc.getDepthAttachmentFormat();

    VkImageView depth_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &depth_view));


    array attachment_views {
        depth_view,
    };

    VkFramebufferCreateInfo fb_info;
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.pNext = nullptr;
    fb_info.flags = 0;
    fb_info.renderPass = render_pass;
    fb_info.attachmentCount = static_cast<uint32_t>(attachment_views.size());
    fb_info.pAttachments = attachment_views.data();
    fb_info.width = fb_width;
    fb_info.height = fb_height;
    fb_info.layers = 1;

    VkFramebuffer hdl;
    REQ_VK(dev.dt.createFramebuffer(dev.hdl, &fb_info, nullptr, &hdl));

    return ShadowFramebuffer {
        std::move(depth),
        depth_view,
        hdl,
    };
}

static void makeFrame(const Device &dev, MemoryAllocator &alloc,
                      uint32_t fb_width, uint32_t fb_height,
                      uint32_t max_views, uint32_t max_instances,
                      VkRenderPass render_pass,
                      VkRenderPass shadow_pass,
                      VkDescriptorSet cull_set,
                      VkDescriptorSet draw_set,
                      Frame *dst)
{
    auto fb = makeFramebuffer(dev, alloc, fb_width, fb_height, render_pass);
    auto shadow_fb = makeShadowFramebuffer(dev, alloc, InternalConfig::shadowMapSize, InternalConfig::shadowMapSize, shadow_pass);

    VkCommandPool cmd_pool = makeCmdPool(dev, dev.gfxQF);

    int64_t buffer_offsets[5];
    int64_t buffer_sizes[6] = {
        (int64_t)sizeof(PackedViewData) * (max_views + 1),
        (int64_t)sizeof(uint32_t),
        (int64_t)sizeof(PackedInstanceData) * max_instances,
        (int64_t)sizeof(DrawCmd) * max_instances * 10,
        (int64_t)sizeof(DrawMaterialData) * max_instances * 10,
        (int64_t)sizeof(DirectionalLight) * InternalConfig::maxLights
    };
    int64_t num_render_input_bytes = utils::computeBufferOffsets(
        buffer_sizes, buffer_offsets, 256);

    HostBuffer view_staging = alloc.makeStagingBuffer(sizeof(PackedViewData));
    HostBuffer light_staging = alloc.makeStagingBuffer(sizeof(DirectionalLight) * InternalConfig::maxLights);
    // HostBuffer shadow_staging = alloc.makeStagingBuffer(sizeof(ShadowViewData));

    LocalBuffer render_input = *alloc.makeLocalBuffer(num_render_input_bytes);

    std::array<VkWriteDescriptorSet, 8> desc_updates;

    VkDescriptorBufferInfo view_info;
    view_info.buffer = render_input.buffer;
    view_info.offset = 0;
    view_info.range = buffer_sizes[0];

    //DescHelper::uniform(desc_updates[0], cull_set, &view_info, 0);
    DescHelper::storage(desc_updates[0], draw_set, &view_info, 0);

    VkDescriptorBufferInfo instance_info;
    instance_info.buffer = render_input.buffer;
    instance_info.offset = buffer_offsets[1];
    instance_info.range = buffer_sizes[2];

    DescHelper::storage(desc_updates[1], cull_set, &instance_info, 1);
    DescHelper::storage(desc_updates[2], draw_set, &instance_info, 1);

    VkDescriptorBufferInfo drawcount_info;
    drawcount_info.buffer = render_input.buffer;
    drawcount_info.offset = buffer_offsets[0];
    drawcount_info.range = buffer_sizes[1];

    DescHelper::storage(desc_updates[3], cull_set, &drawcount_info, 2);

    VkDescriptorBufferInfo draw_info;
    draw_info.buffer = render_input.buffer;
    draw_info.offset = buffer_offsets[2];
    draw_info.range = buffer_sizes[3];

    DescHelper::storage(desc_updates[4], cull_set, &draw_info, 3);

    VkDescriptorBufferInfo draw_mat_info;
    draw_mat_info.buffer = render_input.buffer;
    draw_mat_info.offset = buffer_offsets[3];
    draw_mat_info.range = buffer_sizes[4];

    DescHelper::storage(desc_updates[5], cull_set, &draw_mat_info, 4);
    DescHelper::storage(desc_updates[6], draw_set, &draw_mat_info, 2);

#if 1
    VkDescriptorBufferInfo light_data_info;
    light_data_info.buffer = render_input.buffer;
    light_data_info.offset = buffer_offsets[4];
    light_data_info.range = buffer_sizes[5];

    DescHelper::storage(desc_updates[7], draw_set, &light_data_info, 3);
#endif

    DescHelper::update(dev, desc_updates.data(), desc_updates.size());

    new (dst) Frame {
        std::move(fb),
        std::move(shadow_fb),
        cmd_pool,
        makeCmdBuffer(dev, cmd_pool),
        makeFence(dev, true),
        makeBinarySemaphore(dev),
        makeBinarySemaphore(dev),
        std::move(view_staging),
        std::move(light_staging),
        std::move(render_input),
        0,
        sizeof(PackedViewData),
        uint32_t(buffer_offsets[2]),
        uint32_t(buffer_offsets[0]),
        uint32_t(buffer_offsets[1]),
        (uint32_t)buffer_offsets[4],
        max_instances * 10,
        cull_set,
        draw_set,
    };
}

static array<VkClearValue, 2> makeClearValues()
{
    VkClearValue color_clear;
    color_clear.color = {{0.f, 0.f, 0.f, 1.f}};

    VkClearValue depth_clear;
    depth_clear.depthStencil = {0.f, 0};

    return {
        color_clear,
        depth_clear,
    };
}

static void imguiVkCheck(VkResult res)
{
    REQ_VK(res);
}

static VkRenderPass makeImGuiRenderPass(const Device &dev,
                                        VkFormat color_fmt,
                                        VkFormat depth_fmt)
{
    array<VkAttachmentDescription, 2> attachment_descs {{
        {
            0,
            color_fmt,
            VK_SAMPLE_COUNT_1_BIT,
            VK_ATTACHMENT_LOAD_OP_LOAD,
            VK_ATTACHMENT_STORE_OP_STORE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        },
        {
            0,
            depth_fmt,
            VK_SAMPLE_COUNT_1_BIT,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        },
    }};

    array<VkAttachmentReference, 2> attachment_refs {{
        {
            0,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        },
        {
            1,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        },
    }};
    
    VkSubpassDescription subpass_desc {};
    subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_desc.colorAttachmentCount = attachment_refs.size() - 1;
    subpass_desc.pColorAttachments = attachment_refs.data();
    subpass_desc.pDepthStencilAttachment = &attachment_refs.back();

    VkRenderPassCreateInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.flags = 0;
    render_pass_info.attachmentCount = attachment_descs.size();
    render_pass_info.pAttachments = attachment_descs.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass_desc;
    render_pass_info.dependencyCount = 0;
    render_pass_info.pDependencies = nullptr;

    VkRenderPass render_pass;
    REQ_VK(dev.dt.createRenderPass(dev.hdl, &render_pass_info, nullptr,
                                   &render_pass));

    return render_pass;
}

static PFN_vkVoidFunction imguiVKLookup(const char *fname,
                                        void *user_data)
{
    auto data = (ImGUIVkLookupData *)user_data;

    auto addr = data->getDevAddr(data->dev, fname);

    if (!addr) {
        addr = data->getInstAddr(data->inst, fname);
    }

    if (!addr) {
        FATAL("Failed to load ImGUI vulkan function: %s", fname);
    }

    return addr;
}

static ImGuiRenderState imguiInit(GLFWwindow *window, const Device &dev,
                                  const Backend &backend, VkQueue ui_queue,
                                  VkPipelineCache pipeline_cache,
                                  VkFormat color_fmt,
                                  VkFormat depth_fmt)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    auto font_path = std::filesystem::path(STRINGIFY(VIEWER_DATA_DIR)) /
        "font.ttf";

    float scale_factor;
    {
        float x_scale, y_scale;
        glfwGetWindowContentScale(window, &x_scale, &y_scale);
        assert(x_scale == y_scale);

        scale_factor = x_scale;
    }

    float scaled_font_size = 16.f * scale_factor;
    io.Fonts->AddFontFromFileTTF(font_path.c_str(), scaled_font_size);

    auto &style = ImGui::GetStyle();
    style.ScaleAllSizes(scale_factor);

    ImGui_ImplGlfw_InitForVulkan(window, true);

    // Taken from imgui/examples/example_glfw_vulkan/main.cpp
    VkDescriptorPool desc_pool;
    {
        VkDescriptorPoolSize pool_sizes[] = {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 },
        };
        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
        pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
        pool_info.pPoolSizes = pool_sizes;
        REQ_VK(dev.dt.createDescriptorPool(dev.hdl,
            &pool_info, nullptr, &desc_pool));
    }

    ImGui_ImplVulkan_InitInfo vk_init = {};
    vk_init.Instance = backend.hdl;
    vk_init.PhysicalDevice = dev.phy;
    vk_init.Device = dev.hdl;
    vk_init.QueueFamily = dev.gfxQF;
    vk_init.Queue = ui_queue;
    vk_init.PipelineCache = pipeline_cache;
    vk_init.DescriptorPool = desc_pool;
    vk_init.MinImageCount = InternalConfig::numFrames;
    vk_init.ImageCount = InternalConfig::numFrames;
    vk_init.CheckVkResultFn = imguiVkCheck;

    VkRenderPass imgui_renderpass = makeImGuiRenderPass(dev, color_fmt,
                                                        depth_fmt);

    ImGUIVkLookupData lookup_data {
        dev.dt.getDeviceProcAddr,
        dev.hdl,
        backend.dt.getInstanceProcAddr,
        backend.hdl,
    };
    ImGui_ImplVulkan_LoadFunctions(imguiVKLookup, &lookup_data);
    ImGui_ImplVulkan_Init(&vk_init, imgui_renderpass);

    VkCommandPool tmp_pool = makeCmdPool(dev, dev.gfxQF);
    VkCommandBuffer tmp_cmd = makeCmdBuffer(dev, tmp_pool);
    VkCommandBufferBeginInfo tmp_begin {};
    tmp_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    REQ_VK(dev.dt.beginCommandBuffer(tmp_cmd, &tmp_begin));
    ImGui_ImplVulkan_CreateFontsTexture(tmp_cmd);
    REQ_VK(dev.dt.endCommandBuffer(tmp_cmd));

    VkFence tmp_fence = makeFence(dev);

    VkSubmitInfo tmp_submit {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        0,
        nullptr,
        nullptr,
        1,
        &tmp_cmd,
        0,
        nullptr,
    };

    REQ_VK(dev.dt.queueSubmit(ui_queue, 1, &tmp_submit, tmp_fence));
    waitForFenceInfinitely(dev, tmp_fence);

    dev.dt.destroyFence(dev.hdl, tmp_fence, nullptr);
    dev.dt.destroyCommandPool(dev.hdl, tmp_pool, nullptr);

    return {
        desc_pool,
        imgui_renderpass,
    };
}

static EngineInterop setupEngineInterop(MemoryAllocator &alloc,
                                        uint32_t num_worlds,
                                        uint32_t max_views_per_world,
                                        uint32_t max_instances_per_world,
                                        uint32_t render_width,
                                        uint32_t render_height)
{
    int64_t buffer_offsets[3];
    int64_t buffer_sizes[4] = {
        (int64_t)sizeof(shader::PackedInstanceData) *
            num_worlds * max_instances_per_world,
        (int64_t)sizeof(shader::PackedViewData) *
            num_worlds * max_views_per_world,
        (int64_t)sizeof(uint32_t) * num_worlds,
        (int64_t)sizeof(uint32_t) * num_worlds,
    };

    int64_t num_staging_bytes = utils::computeBufferOffsets(
        buffer_sizes, buffer_offsets, 256);

    HostBuffer staging = alloc.makeStagingBuffer(num_staging_bytes);
    char *staging_base = (char *)staging.ptr;

    InstanceData *instance_base =
        (InstanceData *)staging_base;

    PerspectiveCameraData *view_base = 
        (PerspectiveCameraData *)(staging_base + buffer_offsets[0]);

    InstanceData **world_instances = (InstanceData **)malloc(
        sizeof(InstanceData *) * num_worlds);

    PerspectiveCameraData **world_views = (PerspectiveCameraData **)malloc(
        sizeof(PerspectiveCameraData *) * num_worlds);

    for (CountT i = 0; i < (CountT)num_worlds; i++) {
        world_instances[i] = instance_base + i * max_instances_per_world;
        world_views[i] = view_base + i * max_views_per_world;
    }

    RendererInterface iface {
        .views = world_views,
        .numViews = (uint32_t *)(staging_base + buffer_offsets[1]),
        .instances = world_instances,
        .numInstances = (uint32_t *)(staging_base + buffer_offsets[2]),
        .renderWidth = (int32_t)render_width,
        .renderHeight = (int32_t)render_height,
    };

    return EngineInterop {
        std::move(staging),
        RendererBridge { iface },
        uint32_t(buffer_offsets[0]),
        max_views_per_world,
        max_instances_per_world,
    };
}

Renderer::Renderer(uint32_t gpu_id,
                   uint32_t img_width,
                   uint32_t img_height,
                   uint32_t num_worlds,
                   uint32_t max_views_per_world,
                   uint32_t max_instances_per_world)
    : backend(initializeBackend()),
      window(makeWindow(backend, img_width, img_height)),
      dev(backend.initDevice(
#ifdef MADRONA_CUDA_SUPPORT
          getVkUUIDFromCudaID(gpu_id),
#else
          gpu_id,
#endif
          window.surface)),
      alloc(dev, backend),
      render_queue_(makeGFXQueue(dev, 0)),
      transfer_queue_(makeTransferQueue(dev, 0)),
      compute_queue_(makeComputeQueue(dev, 0)),
      transfer_wrapper_(transfer_queue_, false),
      present_wrapper_(render_queue_, false),
      fb_width_(img_width),
      fb_height_(img_height),
      fb_clear_(makeClearValues()),
      present_(backend, dev, window,
               InternalConfig::numFrames, true),
      pipeline_cache_(getPipelineCache(dev)),
      repeat_sampler_(
          makeImmutableSampler(dev, VK_SAMPLER_ADDRESS_MODE_REPEAT)),
      clamp_sampler_(
          makeImmutableSampler(dev, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)),
      render_pass_(makeRenderPass(dev, alloc.getColorAttachmentFormat(),
                                  alloc.getDepthAttachmentFormat())),
      shadow_pass_(makeShadowRenderPass(dev, alloc.getDepthAttachmentFormat())),
      imgui_render_state_(imguiInit(window.platformWindow, dev, backend,
                                 render_queue_, pipeline_cache_,
                                 alloc.getColorAttachmentFormat(),
                                 alloc.getDepthAttachmentFormat())),
      instance_cull_(makeCullPipeline(dev, pipeline_cache_,
                                      InternalConfig::numFrames)),
      object_draw_(makeDrawPipeline(dev, pipeline_cache_, render_pass_,
                                    repeat_sampler_, clamp_sampler_,
                                    InternalConfig::numFrames)),
      object_shadow_draw_(makeShadowDrawPipeline(dev, pipeline_cache_, shadow_pass_,
                                    repeat_sampler_, clamp_sampler_,
                                    InternalConfig::numFrames)),
      asset_desc_pool_cull_(dev, instance_cull_.shaders, 1, 1),
      asset_desc_pool_draw_(dev, object_draw_.shaders, 1, 1),
      asset_set_cull_(asset_desc_pool_cull_.makeSet()),
      asset_set_draw_(asset_desc_pool_draw_.makeSet()),
      load_cmd_pool_(makeCmdPool(dev, dev.gfxQF)),
      load_cmd_(makeCmdBuffer(dev, load_cmd_pool_)),
      load_fence_(makeFence(dev)),
      engine_interop_(setupEngineInterop(
          alloc, num_worlds,
          max_views_per_world, max_instances_per_world,
          fb_width_, fb_height_)),
      lights_(InternalConfig::maxLights),
      cur_frame_(0),
      frames_(InternalConfig::numFrames),
      loaded_assets_(0)
{
    for (int i = 0; i < (int)frames_.size(); i++) {
        makeFrame(dev, alloc, fb_width_, fb_height_,
                  max_views_per_world, max_instances_per_world,
                  render_pass_,
                  shadow_pass_,
                  instance_cull_.descPool.makeSet(),
                  object_draw_.descPool.makeSet(),
                  &frames_[i]);
    }
}

Renderer::~Renderer()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    loaded_assets_.clear();

    for (Frame &f : frames_) {
        dev.dt.destroySemaphore(dev.hdl, f.swapchainReady, nullptr);
        dev.dt.destroySemaphore(dev.hdl, f.renderFinished, nullptr);

        dev.dt.destroyFence(dev.hdl, f.cpuFinished, nullptr);
        dev.dt.destroyCommandPool(dev.hdl, f.cmdPool, nullptr);

        dev.dt.destroyFramebuffer(dev.hdl, f.fb.hdl, nullptr);
        dev.dt.destroyImageView(dev.hdl, f.fb.colorView, nullptr);
        dev.dt.destroyImageView(dev.hdl, f.fb.depthView, nullptr);
    }

    dev.dt.destroyFence(dev.hdl, load_fence_, nullptr);
    dev.dt.destroyCommandPool(dev.hdl, load_cmd_pool_, nullptr);

    dev.dt.destroyPipeline(dev.hdl, object_draw_.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, object_draw_.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, instance_cull_.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, instance_cull_.layout, nullptr);

    dev.dt.destroyRenderPass(dev.hdl, imgui_render_state_.renderPass, nullptr);
    dev.dt.destroyDescriptorPool(
        dev.hdl, imgui_render_state_.descPool, nullptr);

    dev.dt.destroyRenderPass(dev.hdl, render_pass_, nullptr);

    dev.dt.destroySampler(dev.hdl, clamp_sampler_, nullptr);
    dev.dt.destroySampler(dev.hdl, repeat_sampler_, nullptr);

    dev.dt.destroyPipelineCache(dev.hdl, pipeline_cache_, nullptr);

    present_.destroy(dev);

    backend.dt.destroySurfaceKHR(backend.hdl, window.surface, nullptr);
    glfwDestroyWindow(window.platformWindow);
}

CountT Renderer::loadObjects(Span<const imp::SourceObject> src_objs, Span<const imp::SourceMaterial> src_mats)
{
    using namespace imp;
    using namespace math;

    assert(loaded_assets_.size() == 0);

    int64_t num_total_vertices = 0;
    int64_t num_total_indices = 0;
    int64_t num_total_meshes = 0;

    for (const SourceObject &obj : src_objs) {
        num_total_meshes += obj.meshes.size();

        for (const SourceMesh &mesh : obj.meshes) {
            if (mesh.faceCounts != nullptr) {
                FATAL("Render mesh isn't triangular");
            }

            num_total_vertices += mesh.numVertices;
            num_total_indices += mesh.numFaces * 3;
        }
    }

    int64_t num_total_objs = src_objs.size();

    int64_t buffer_offsets[4];
    int64_t buffer_sizes[5] = {
        (int64_t)sizeof(ObjectData) * num_total_objs,
        (int64_t)sizeof(MeshData) * num_total_meshes,
        (int64_t)sizeof(PackedVertex) * num_total_vertices,
        (int64_t)sizeof(uint32_t) * num_total_indices,
        (int64_t)sizeof(MaterialData) * src_mats.size()
    };
    int64_t num_asset_bytes = utils::computeBufferOffsets(
        buffer_sizes, buffer_offsets, 256);

    HostBuffer staging = alloc.makeStagingBuffer(num_asset_bytes);
    char *staging_ptr = (char *)staging.ptr;
    ObjectData *obj_ptr = (ObjectData *)staging_ptr;
    MeshData *mesh_ptr = 
        (MeshData *)(staging_ptr + buffer_offsets[0]);
    PackedVertex *vertex_ptr =
        (PackedVertex *)(staging_ptr + buffer_offsets[1]);
    uint32_t *indices_ptr =
        (uint32_t *)(staging_ptr + buffer_offsets[2]);
    MaterialData *materials_ptr =
        (MaterialData *)(staging_ptr + buffer_offsets[3]);

    int32_t mesh_offset = 0;
    int32_t vertex_offset = 0;
    int32_t index_offset = 0;
    for (const SourceObject &obj : src_objs) {
        *obj_ptr++ = ObjectData {
            .meshOffset = mesh_offset,
            .numMeshes = (int32_t)obj.meshes.size(),
        };

        for (const SourceMesh &mesh : obj.meshes) {
            int32_t num_mesh_verts = (int32_t)mesh.numVertices;
            int32_t num_mesh_indices = (int32_t)mesh.numFaces * 3;

            mesh_ptr[mesh_offset++] = MeshData {
                .vertexOffset = vertex_offset,
                .numVertices = num_mesh_verts,
                .indexOffset = index_offset,
                .numIndices = num_mesh_indices,
                .materialIndex = (int32_t)mesh.materialIDX
            };

            // Compute new normals
            auto new_normals = Optional<HeapArray<Vector3>>::none();
            if (!mesh.normals) {
                new_normals.emplace(num_mesh_verts);

                for (int64_t vert_idx = 0; vert_idx < num_mesh_verts;
                     vert_idx++) {
                    (*new_normals)[vert_idx] = Vector3::zero();
                }

                for (CountT face_idx = 0; face_idx < (CountT)mesh.numFaces;
                     face_idx++) {
                    CountT base_idx = face_idx * 3;
                    uint32_t i0 = mesh.indices[base_idx];
                    uint32_t i1 = mesh.indices[base_idx + 1];
                    uint32_t i2 = mesh.indices[base_idx + 2];

                    Vector3 v0 = mesh.positions[i0];
                    Vector3 v1 = mesh.positions[i1];
                    Vector3 v2 = mesh.positions[i2];

                    Vector3 e0 = v1 - v0;
                    Vector3 e1 = v2 - v0;

                    Vector3 face_normal = cross(e0, e1);
                    float face_len = face_normal.length();
                    assert(face_len != 0);
                    face_normal /= face_len;

                    (*new_normals)[i0] += face_normal;
                    (*new_normals)[i1] += face_normal;
                    (*new_normals)[i2] += face_normal;
                }

                for (int64_t vert_idx = 0; vert_idx < num_mesh_verts;
                     vert_idx++) {
                    (*new_normals)[vert_idx] =
                        normalize((*new_normals)[vert_idx]);
                }
            }

            for (int32_t i = 0; i < num_mesh_verts; i++) {
                Vector3 pos = mesh.positions[i];
                Vector3 normal = mesh.normals ?
                    mesh.normals[i] : (*new_normals)[i];
                Vector4 tangent_sign;
                // FIXME: use mikktspace at import time
                if (mesh.tangentAndSigns != nullptr) {
                    tangent_sign = mesh.tangentAndSigns[i];
                } else {
                    Vector3 a, b;
                    normal.frame(&a, &b);
                    tangent_sign = {
                        a.x,
                        a.y,
                        a.z,
                        1.f,
                    };
                }
                Vector2 uv = mesh.uvs ? mesh.uvs[i] : Vector2 { 0, 0 };

                Vector3 encoded_normal_tangent =
                    encodeNormalTangent(normal, tangent_sign);

                vertex_ptr[vertex_offset++] = PackedVertex {
                    Vector4 {
                        pos.x,
                        pos.y,
                        pos.z,
                        encoded_normal_tangent.x,
                    },
                    Vector4 {
                        encoded_normal_tangent.y,
                        encoded_normal_tangent.z,
                        uv.x,
                        uv.y,
                    },
                };
            }

            memcpy(indices_ptr + index_offset,
                   mesh.indices, sizeof(uint32_t) * num_mesh_indices);

            index_offset += num_mesh_indices;
        }
    }

    uint32_t mat_idx = 0;
    for (const SourceMaterial &mat : src_mats) {
        materials_ptr[mat_idx++].color = mat.color;
    }

    staging.flush(dev);

    LocalBuffer asset_buffer = *alloc.makeLocalBuffer(num_asset_bytes);
    GPURunUtil gpu_run {
        load_cmd_pool_,
        load_cmd_,
        render_queue_, // FIXME
        load_fence_
    };

    gpu_run.begin(dev);

    VkBufferCopy buffer_copy {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = (VkDeviceSize)num_asset_bytes,
    };

    dev.dt.cmdCopyBuffer(load_cmd_, staging.buffer, asset_buffer.buffer,
                         1, &buffer_copy);
    

    gpu_run.submit(dev);

    std::array<VkWriteDescriptorSet, 4> desc_updates;

    VkDescriptorBufferInfo obj_info;
    obj_info.buffer = asset_buffer.buffer;
    obj_info.offset = 0;
    obj_info.range = buffer_sizes[0];

    DescHelper::storage(desc_updates[0], asset_set_cull_, &obj_info, 0);

    VkDescriptorBufferInfo mesh_info;
    mesh_info.buffer = asset_buffer.buffer;
    mesh_info.offset = buffer_offsets[0];
    mesh_info.range = buffer_sizes[1];

    DescHelper::storage(desc_updates[1], asset_set_cull_, &mesh_info, 1);

    VkDescriptorBufferInfo vert_info;
    vert_info.buffer = asset_buffer.buffer;
    vert_info.offset = buffer_offsets[1];
    vert_info.range = buffer_sizes[2];

    DescHelper::storage(desc_updates[2], asset_set_draw_, &vert_info, 0);

    VkDescriptorBufferInfo mat_info;
    mat_info.buffer = asset_buffer.buffer;
    mat_info.offset = buffer_offsets[3];
    mat_info.range = buffer_sizes[4];

    DescHelper::storage(desc_updates[3], asset_set_draw_, &mat_info, 1);

    DescHelper::update(dev, desc_updates.data(), desc_updates.size());

    AssetData asset_data {
        std::move(asset_buffer),
        (uint32_t)buffer_offsets[2],
    };

    loaded_assets_.emplace_back(std::move(asset_data));

    return 0;
}

void Renderer::configureLighting(Span<const LightConfig> lights)
{
    for (int i = 0; i < lights.size(); ++i) {
        lights_.insert(i, DirectionalLight{ 
            math::Vector4{lights[i].dir.x, lights[i].dir.y, lights[i].dir.z, 1.0f }, 
            math::Vector4{lights[i].color.x, lights[i].color.y, lights[i].color.z, 1.0f}
        });
    }
}

void Renderer::waitUntilFrameReady()
{
    Frame &frame = frames_[cur_frame_];
    // Wait until frame using this slot has finished
    REQ_VK(dev.dt.waitForFences(dev.hdl, 1, &frame.cpuFinished, VK_TRUE,
                                UINT64_MAX));
}

void Renderer::startFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
}

static void packView(const Device &dev,
                     HostBuffer &view_staging_buffer,
                     const ViewerCam &cam,
                     uint32_t fb_width, uint32_t fb_height)
{
    PackedViewData *staging = (PackedViewData *)view_staging_buffer.ptr;

    math::Quat rotation =
        math::Quat::fromBasis(cam.right, cam.view, cam.up).inv();

    float fov_scale = 1.f / tanf(math::toRadians(cam.fov * 0.5f));
    float aspect = float(fb_width) / float(fb_height);

    float x_scale = fov_scale / aspect;
    float y_scale = -fov_scale;

    math::Vector4 d0 {
        cam.position.x,
        cam.position.y,
        cam.position.z,
        rotation.w,
    };

    math::Vector4 d1 {
        rotation.x,
        rotation.y,
        rotation.z,
        x_scale,
    };

    math::Vector4 d2 {
        y_scale,
        0.001f,
        0.f,
        0.f,
    };

    staging->data[0] = d0;
    staging->data[1] = d1;
    staging->data[2] = d2;

    view_staging_buffer.flush(dev);
}

static void packLighting(const Device &dev,
                         HostBuffer &light_staging_buffer,
                         const HeapArray<DirectionalLight> &lights)
{
    DirectionalLight *staging = (DirectionalLight *)light_staging_buffer.ptr;
    memcpy(staging, lights.data(), sizeof(DirectionalLight) * InternalConfig::maxLights);
    light_staging_buffer.flush(dev);
}

void Renderer::render(const ViewerCam &cam,
                      const FrameConfig &cfg)
{
    Frame &frame = frames_[cur_frame_];
    uint32_t swapchain_idx = present_.acquireNext(dev, frame.swapchainReady);

    // Need to flush engine input state
    engine_interop_.renderInputStaging.flush(dev);

    REQ_VK(dev.dt.resetCommandPool(dev.hdl, frame.cmdPool, 0));
    VkCommandBuffer draw_cmd = frame.drawCmd;

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(draw_cmd, &begin_info));

    packView(dev, frame.viewStaging, cam, fb_width_, fb_height_);
    VkBufferCopy view_copy {
        .srcOffset = 0,
        .dstOffset = frame.cameraViewOffset,
        .size = sizeof(PackedViewData)
    };

    dev.dt.cmdCopyBuffer(draw_cmd, frame.viewStaging.buffer,
                         frame.renderInput.buffer,
                         1, &view_copy);

    packLighting(dev, frame.lightStaging, lights_);
    VkBufferCopy light_copy {
        .srcOffset = 0,
        .dstOffset = frame.lightOffset,
        .size = sizeof(DirectionalLight) * InternalConfig::maxLights
    };

    dev.dt.cmdCopyBuffer(draw_cmd, frame.lightStaging.buffer,
                         frame.renderInput.buffer,
                         1, &light_copy);

    dev.dt.cmdFillBuffer(draw_cmd, frame.renderInput.buffer,
                         frame.drawCountOffset, sizeof(uint32_t), 0);

    uint32_t num_sim_views =
        engine_interop_.bridge.iface.numViews[cfg.worldIDX];
    if (num_sim_views > 0) {
        VkDeviceSize world_view_byte_offset = engine_interop_.viewBaseOffset +
            cfg.worldIDX * engine_interop_.maxViewsPerWorld *
            sizeof(PackedViewData);

        VkBufferCopy view_data_copy {
            .srcOffset = world_view_byte_offset,
            .dstOffset = frame.simViewOffset,
            .size = sizeof(PackedViewData) * num_sim_views,
        };

        dev.dt.cmdCopyBuffer(draw_cmd,
                             engine_interop_.renderInputStaging.buffer,
                             frame.renderInput.buffer,
                             1, &view_data_copy);
    }

    uint32_t num_instances =
        engine_interop_.bridge.iface.numInstances[cfg.worldIDX];

    if (num_instances > 0) {
        VkDeviceSize world_instance_byte_offset = sizeof(PackedInstanceData) *
            cfg.worldIDX * engine_interop_.maxInstancesPerWorld;

        VkBufferCopy instance_copy = {
            .srcOffset = world_instance_byte_offset,
            .dstOffset = frame.instanceOffset,
            .size = sizeof(PackedInstanceData) * num_instances,
        };

        dev.dt.cmdCopyBuffer(draw_cmd,
                             engine_interop_.renderInputStaging.buffer,
                             frame.renderInput.buffer,
                             1, &instance_copy);

        VkMemoryBarrier copy_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };

        dev.dt.cmdPipelineBarrier(draw_cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 1, &copy_barrier, 0, nullptr, 0, nullptr);

        dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               instance_cull_.hdls[0]);

        std::array cull_descriptors {
            frame.cullShaderSet,
            asset_set_cull_,
        };

        dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     instance_cull_.layout, 0,
                                     cull_descriptors.size(),
                                     cull_descriptors.data(),
                                     0, nullptr);

        CullPushConst cull_push_const {
            num_instances,
        };

        dev.dt.cmdPushConstants(draw_cmd, instance_cull_.layout,
                                VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                sizeof(CullPushConst), &cull_push_const);

        uint32_t num_workgroups = utils::divideRoundUp(num_instances, 32_u32);
        dev.dt.cmdDispatch(draw_cmd, num_workgroups, 1, 1);

        VkMemoryBarrier cull_draw_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
        };

        dev.dt.cmdPipelineBarrier(draw_cmd,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                  0, 1, &cull_draw_barrier, 0, nullptr,
                                  0, nullptr);
    } else {
        VkMemoryBarrier copy_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask =
                VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                VK_ACCESS_UNIFORM_READ_BIT,
        };

        dev.dt.cmdPipelineBarrier(draw_cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 1, &copy_barrier, 0, nullptr, 0, nullptr);
    }

    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           object_draw_.hdls[0]);

    std::array draw_descriptors {
        frame.drawShaderSet,
        asset_set_draw_,
    };

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 object_draw_.layout, 0,
                                 draw_descriptors.size(),
                                 draw_descriptors.data(),
                                 0, nullptr);

    DrawPushConst draw_const {
        (uint32_t)cfg.viewIDX,
    };

    dev.dt.cmdPushConstants(draw_cmd, object_draw_.layout,
                            VK_SHADER_STAGE_VERTEX_BIT |
                            VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                            sizeof(DrawPushConst), &draw_const);

    dev.dt.cmdBindIndexBuffer(draw_cmd, loaded_assets_[0].buf.buffer,
                              loaded_assets_[0].idxBufferOffset,
                              VK_INDEX_TYPE_UINT32);

    VkViewport viewport {
        0,
        0,
        (float)fb_width_,
        (float)fb_height_,
        0.f,
        1.f,
    };

    dev.dt.cmdSetViewport(draw_cmd, 0, 1, &viewport);

    VkRect2D scissor {
        { 0, 0 },
        { fb_width_, fb_height_ },
    };

    dev.dt.cmdSetScissor(draw_cmd, 0, 1, &scissor);


    VkRenderPassBeginInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.renderPass = render_pass_;
    render_pass_info.framebuffer = frame.fb.hdl;
    render_pass_info.clearValueCount = fb_clear_.size();
    render_pass_info.pClearValues = fb_clear_.data();
    render_pass_info.renderArea.offset = {
        0, 0,
    };
    render_pass_info.renderArea.extent = {
        fb_width_, fb_height_,
    };

    dev.dt.cmdBeginRenderPass(draw_cmd, &render_pass_info,
                              VK_SUBPASS_CONTENTS_INLINE);

    dev.dt.cmdDrawIndexedIndirect(draw_cmd,
                                  frame.renderInput.buffer,
                                  frame.drawCmdOffset,
                                  num_instances,
                                  sizeof(DrawCmd));

    dev.dt.cmdEndRenderPass(draw_cmd);

    render_pass_info.renderPass = imgui_render_state_.renderPass;
    dev.dt.cmdBeginRenderPass(draw_cmd, &render_pass_info,
                              VK_SUBPASS_CONTENTS_INLINE);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), draw_cmd);

    dev.dt.cmdEndRenderPass(draw_cmd);

    VkImage swapchain_img = present_.getImage(swapchain_idx);

    array<VkImageMemoryBarrier, 2> blit_prepare {{
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            swapchain_img,
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            },
        },
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            frame.fb.colorAttachment.image,
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            },
        }
    }};

    dev.dt.cmdPipelineBarrier(draw_cmd,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT |
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr, 0, nullptr,
        blit_prepare.size(), blit_prepare.data());

    VkImageBlit blit_region {
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        {
            { 0, 0, 0 }, 
            { (int32_t)fb_width_, (int32_t)fb_height_, 1 },
        },
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        {
            { 0, 0, 0 }, 
            { (int32_t)fb_width_, (int32_t)fb_height_, 1 },
        },
    };

    dev.dt.cmdBlitImage(draw_cmd,
                        frame.fb.colorAttachment.image,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        swapchain_img,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &blit_region,
                        VK_FILTER_NEAREST);

    VkImageMemoryBarrier swapchain_prepare {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        0,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        swapchain_img,
        {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0, 1, 0, 1
        },
    };

    dev.dt.cmdPipelineBarrier(draw_cmd,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                              0,
                              0, nullptr, 0, nullptr,
                              1, &swapchain_prepare);

    REQ_VK(dev.dt.endCommandBuffer(draw_cmd));

    VkPipelineStageFlags swapchain_wait_flag =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo gfx_submit {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        1,
        &frame.swapchainReady,
        &swapchain_wait_flag,
        1,
        &draw_cmd,
        1,
        &frame.renderFinished,
    };

    REQ_VK(dev.dt.resetFences(dev.hdl, 1, &frame.cpuFinished));
    REQ_VK(dev.dt.queueSubmit(render_queue_, 1, &gfx_submit,
                              frame.cpuFinished));

    present_.present(dev, swapchain_idx, present_wrapper_,
                     1, &frame.renderFinished);

    cur_frame_ = (cur_frame_ + 1) % frames_.size();
}

void Renderer::waitForIdle()
{
    dev.dt.deviceWaitIdle(dev.hdl);
}

const render::RendererBridge & Renderer::getBridgeRef() const
{
    return engine_interop_.bridge;
}

}
