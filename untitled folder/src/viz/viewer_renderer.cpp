#include "viewer_renderer.hpp"
#include <chrono>
#include <madrona/render/shader_compiler.hpp>

#include "backends/imgui_impl_vulkan.h"
#include "backends/imgui_impl_glfw.h"

#include "../render/asset_utils.hpp"

#include "vk/descriptors.hpp"
#include <iostream>
#include <stdexcept>

#include <filesystem>

#ifdef MADRONA_MACOS
#include <dlfcn.h>
#endif

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

#include "shader.hpp"

#include <fstream>
#include <random>

#include <signal.h>

#include <stb_image.h>
#include <stb_image_write.h>

using namespace std;

using namespace madrona::render;
using namespace madrona::render::vk;
/*
uint32_t findMemoryType(InstanceDispatch& instanceDispatch, VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties2 memProperties;
    instanceDispatch.getPhysicalDeviceMemoryProperties2(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    std::cout << "Failed to find suitable memory type!" << std::endl;
}*/

/*uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    std::cout << "failed to find suitable memory type!" << std::endl;
}*/
uint32_t findMemoryType(madrona::render::vk::InstanceDispatch& instanceDispatch, VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties2 memProperties2{};
    memProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;

    // Use the member function of the InstanceDispatch object
    instanceDispatch.getPhysicalDeviceMemoryProperties2(physicalDevice, &memProperties2);

    auto& memProperties = memProperties2.memoryProperties;

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    std::cout << "failed to find suitable memory type!" << std::endl;
}





namespace madrona::viz {

using Vertex = shader::Vertex;
using PackedVertex = shader::PackedVertex;
using MeshData = shader::MeshData;
using MaterialData = shader::MaterialData;
using ObjectData = shader::ObjectData;
using DrawPushConst = shader::DrawPushConst;
using CullPushConst = shader::CullPushConst;
using DeferredLightingPushConst = shader::DeferredLightingPushConst;
using DrawCmd = shader::DrawCmd;
using DrawData = shader::DrawData;
using PackedInstanceData = shader::PackedInstanceData;
using PackedViewData = shader::PackedViewData;
using ShadowViewData = shader::ShadowViewData;
using DirectionalLight = shader::DirectionalLight;
using SkyData = shader::SkyData;
using DensityLayer = shader::DensityLayer;

namespace InternalConfig {

inline constexpr uint32_t numFrames = 2;
inline constexpr uint32_t initMaxTransforms = 100000;
inline constexpr uint32_t initMaxMatIndices = 100000;
inline constexpr uint32_t shadowMapSize = 4096;
inline constexpr uint32_t maxLights = 10;
inline constexpr uint32_t maxTextures = 100;
inline constexpr VkFormat gbufferFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
inline constexpr VkFormat skyFormatHighp = VK_FORMAT_R32G32B32A32_SFLOAT;
inline constexpr VkFormat skyFormatHalfp = VK_FORMAT_R16G16B16A16_SFLOAT;
inline constexpr VkFormat varianceFormat = VK_FORMAT_R32G32_SFLOAT;
inline constexpr VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

}

struct ImGUIVkLookupData {
    PFN_vkGetDeviceProcAddr getDevAddr;
    VkDevice dev;
    PFN_vkGetInstanceProcAddr getInstAddr;
    VkInstance inst;
    
};

Backend::LoaderLib PresentationState::init()
{
#ifdef MADRONA_MACOS
    Backend::LoaderLib loader_lib = Backend::loadLoaderLib();

    glfwInitVulkanLoader((PFN_vkGetInstanceProcAddr)loader_lib.getEntryFn());
#else
    Backend::LoaderLib loader_lib(nullptr, nullptr);
#endif

    if (!glfwInit()) {
        FATAL("Failed to initialize GLFW");
    }

    return loader_lib;
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
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
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

//Voxelizer changes
static PipelineShaders makeVoxelGenShader(const Device& dev)
{
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(VIEWER_DATA_DIR)) /
        "shaders";

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / "voxel_gen.hlsl").string().c_str(), {},
        {}, { "voxelGen", ShaderStage::Compute });

    StackAlloc tmp_alloc;
    return PipelineShaders(
        dev, tmp_alloc,
        Span<const SPIRVShader>(&spirv, 1),
        {});
}

static Pipeline<1> makeVoxelMeshGenPipeline(const Device& dev,
    VkPipelineCache pipeline_cache,
    CountT num_frames) {
    PipelineShaders shader = makeVoxelGenShader(dev);

    // Push constant
    VkPushConstantRange push_const{
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(shader::VoxelGenPushConst),
    };

    // Layout configuration
    std::array desc_layouts {
        shader.getLayout(0),
    };

    VkPipelineLayoutCreateInfo voxel_gen_layout_info;
    voxel_gen_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    voxel_gen_layout_info.pNext = nullptr;
    voxel_gen_layout_info.flags = 0;
    voxel_gen_layout_info.setLayoutCount =
        static_cast<uint32_t>(desc_layouts.size());
    voxel_gen_layout_info.pSetLayouts = desc_layouts.data();
    voxel_gen_layout_info.pushConstantRangeCount = 1;
    voxel_gen_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout voxel_gen_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &voxel_gen_layout_info, nullptr,
        &voxel_gen_layout));

    std::array<VkComputePipelineCreateInfo, 1> compute_infos;
    compute_infos[0].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_infos[0].pNext = nullptr;
    compute_infos[0].flags = 0;
    compute_infos[0].stage = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, //&subgroup_size,
        VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
        VK_SHADER_STAGE_COMPUTE_BIT,
        shader.getShader(0),
        "voxelGen",
        nullptr,
    };
    compute_infos[0].layout = voxel_gen_layout;
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
            voxel_gen_layout,
            pipelines,
            std::move(desc_pool),
    };
}

static PipelineShaders makeVoxelDrawShaders(
    const Device& dev, VkSampler repeat_sampler, VkSampler clamp_sampler)
{
    (void)repeat_sampler;
    (void)clamp_sampler;

    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(VIEWER_DATA_DIR)) /
        "shaders";

    auto shader_path = (shader_dir / "voxel_draw.hlsl").string();

    ShaderCompiler compiler;
    SPIRVShader vert_spirv = compiler.compileHLSLFileToSPV(
        shader_path.c_str(), {}, {},
        { "vert", ShaderStage::Vertex });

    SPIRVShader frag_spirv = compiler.compileHLSLFileToSPV(
        shader_path.c_str(), {}, {},
        { "frag", ShaderStage::Fragment });
        std::array<SPIRVShader, 2> shaders {
        std::move(vert_spirv),
            std::move(frag_spirv),
    };

    StackAlloc tmp_alloc;
    return PipelineShaders(dev, tmp_alloc, shaders,
        Span<const BindingOverride>({
            BindingOverride {
                1,
                0,
                VK_NULL_HANDLE,
                InternalConfig::maxTextures,
                VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            },
            BindingOverride {
                1,
                1,
                repeat_sampler,
                1,
                0,
            },
            }));
}

static void issueVoxelGen(Device& dev,
    Frame& frame,
    Pipeline<1>& pipeline,
    VkCommandBuffer draw_cmd,
    uint32_t view_idx,
    uint32_t max_views,
    VoxelConfig voxel_config)
{
    const uint32_t num_voxels = voxel_config.xLength * voxel_config.yLength * voxel_config.zLength;

    {
        VkBufferMemoryBarrier compute_prepare = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
           VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT|VK_ACCESS_MEMORY_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            frame.voxelData.buffer,
            0,
            (VkDeviceSize)(num_voxels * sizeof(uint32_t))
        };

        dev.dt.cmdPipelineBarrier(draw_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 1, &compute_prepare, 0, nullptr);
    }

    {
        VkBufferMemoryBarrier compute_prepare = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
           VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            frame.voxelVBO.buffer,
            0,
            (VkDeviceSize)(32 * 4 * 6 * num_voxels)
        };

        dev.dt.cmdPipelineBarrier(draw_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 1, &compute_prepare, 0, nullptr);
    }


    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.hdls[0]);

    shader::VoxelGenPushConst push_const = {voxel_config.xLength,voxel_config.yLength,voxel_config.zLength, 0.8, 9 };

    dev.dt.cmdPushConstants(draw_cmd,
        pipeline.layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(shader::VoxelGenPushConst),
        &push_const);

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.layout, 0, 1, &frame.voxelGenSet, 0, nullptr);

    //uint32_t num_workgroups_x = utils::divideRoundUp(max_views, 32_u32);
    dev.dt.cmdDispatch(draw_cmd, 64, 1, 1);

    {
        VkBufferMemoryBarrier compute_prepare = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            frame.voxelVBO.buffer,
            0,
            (VkDeviceSize)(32 * 4 * 6 * num_voxels)
        };

        dev.dt.cmdPipelineBarrier(draw_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
            0, 0, nullptr, 1, &compute_prepare, 0, nullptr);
    }
    /* {
        VkBufferMemoryBarrier compute_prepare = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            frame.voxelIndexBuffer.buffer,
            0,
            (VkDeviceSize)(6 *6* 4 * 50 * 50 * 50)
        };

        dev.dt.cmdPipelineBarrier(draw_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
            0, 0, nullptr, 1, &compute_prepare, 0, nullptr);
    }*/
}



static VkRenderPass makeRenderPass(const Device &dev,
                                   VkFormat color_fmt,
                                   VkFormat normal_fmt,
                                   VkFormat position_fmt,
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

    attachment_descs.push_back(
        {0, normal_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    attachment_descs.push_back(
        {0, position_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    attachment_descs.push_back(
        {0, depth_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

    
    attachment_refs.push_back(
        {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    attachment_refs.push_back(
        {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    attachment_refs.push_back(
        {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    attachment_refs.push_back(
        {3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

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
                                         VkFormat variance_fmt,
                                         VkFormat depth_fmt)
{
    vector<VkAttachmentDescription> attachment_descs;
    vector<VkAttachmentReference> attachment_refs;

    attachment_descs.push_back(
        {0, variance_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    attachment_descs.push_back(
        {0, depth_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

    attachment_refs.push_back(
        {static_cast<uint32_t>(attachment_refs.size()),
         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    attachment_refs.push_back(
        {static_cast<uint32_t>(attachment_refs.size()),
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

    VkSubpassDescription subpass_desc {};
    subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_desc.colorAttachmentCount = 1;
    subpass_desc.pColorAttachments = &attachment_refs.front();
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

    auto shader_path = (shader_dir / "viewer_draw.hlsl").string();

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
    return PipelineShaders(dev, tmp_alloc, shaders,
        Span<const BindingOverride>({
            BindingOverride {
                2,
                0,
                VK_NULL_HANDLE,
                InternalConfig::maxTextures,
                VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            },
            BindingOverride {
                2,
                1,
                repeat_sampler,
                1,
                0,
            },
        }));
}

static PipelineShaders makeShadowDrawShaders(
    const Device &dev, VkSampler repeat_sampler, VkSampler clamp_sampler)
{
    (void)repeat_sampler;
    (void)clamp_sampler;

    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(VIEWER_DATA_DIR)) /
        "shaders";

    auto shader_path = (shader_dir / "viewer_shadow_draw.hlsl").string();

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
        (shader_dir / "viewer_cull.hlsl").string().c_str(), {},
        {}, { "instanceCull", ShaderStage::Compute });

    StackAlloc tmp_alloc;
    return PipelineShaders(dev, tmp_alloc,
                           Span<const SPIRVShader>(&spirv, 1), {});
}

static PipelineShaders makeShadowGenShader(const Device &dev, VkSampler clamp_sampler)
{
    (void)clamp_sampler;
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(VIEWER_DATA_DIR)) /
        "shaders";

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / "shadow_gen.hlsl").string().c_str(), {},
        {}, { "shadowGen", ShaderStage::Compute });

    StackAlloc tmp_alloc;
    return PipelineShaders(
        dev, tmp_alloc,
        Span<const SPIRVShader>(&spirv, 1), 
        {});
}

static PipelineShaders makeDeferredLightingShader(const Device &dev, VkSampler clamp_sampler)
{
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(VIEWER_DATA_DIR)) /
        "shaders";

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / "viewer_deferred_lighting.hlsl").string().c_str(), {},
        {}, { "lighting", ShaderStage::Compute });

    StackAlloc tmp_alloc;
    return PipelineShaders(
        dev, tmp_alloc,
        Span<const SPIRVShader>(&spirv, 1), 
        Span<const BindingOverride>({BindingOverride{
            0, 9, clamp_sampler, 1, 0 }}));
}

static PipelineShaders makeBlurShader(const Device &dev, VkSampler clamp_sampler)
{
    (void)clamp_sampler;
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(VIEWER_DATA_DIR)) /
        "shaders";

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / "blur.hlsl").string().c_str(), {},
        {}, { "blur", ShaderStage::Compute });

    StackAlloc tmp_alloc;
    return PipelineShaders(
        dev, tmp_alloc,
        Span<const SPIRVShader>(&spirv, 1), 
        {});
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

static void initCommonDrawPipelineInfo(VkPipelineVertexInputStateCreateInfo &vert_info,
                                       VkPipelineInputAssemblyStateCreateInfo &input_assembly_info,
                                       VkPipelineViewportStateCreateInfo &viewport_info,
                                       VkPipelineMultisampleStateCreateInfo &multisample_info,
                                       VkPipelineRasterizationStateCreateInfo &raster_info) {
    // Disable auto vertex assembly
    vert_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vert_info.pNext = nullptr;
    vert_info.flags = 0;
    vert_info.vertexBindingDescriptionCount = 0;
    vert_info.pVertexBindingDescriptions = nullptr;
    vert_info.vertexAttributeDescriptionCount = 0;
    vert_info.pVertexAttributeDescriptions = nullptr;

    // Assembly (standard tri indices)
    input_assembly_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly_info.primitiveRestartEnable = VK_FALSE;

    // Viewport (fully dynamic)
    viewport_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_info.viewportCount = 1;
    viewport_info.pViewports = nullptr;
    viewport_info.scissorCount = 1;
    viewport_info.pScissors = nullptr;

    // Multisample
    multisample_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample_info.sampleShadingEnable = VK_FALSE;
    multisample_info.alphaToCoverageEnable = VK_FALSE;
    multisample_info.alphaToOneEnable = VK_FALSE;

    // Rasterization
    raster_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster_info.depthClampEnable = VK_FALSE;
    raster_info.rasterizerDiscardEnable = VK_FALSE;
    raster_info.polygonMode = VK_POLYGON_MODE_FILL;
    raster_info.cullMode = VK_CULL_MODE_BACK_BIT;
    raster_info.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster_info.depthBiasEnable = VK_FALSE;
    raster_info.lineWidth = 1.0f;
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

    VkPipelineVertexInputStateCreateInfo vert_info {};
    VkPipelineInputAssemblyStateCreateInfo input_assembly_info {};
    VkPipelineViewportStateCreateInfo viewport_info {};
    VkPipelineMultisampleStateCreateInfo multisample_info {};
    VkPipelineRasterizationStateCreateInfo raster_info {};

    initCommonDrawPipelineInfo(vert_info, input_assembly_info, 
        viewport_info, multisample_info, raster_info);

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

    array<VkPipelineColorBlendAttachmentState, 3> blend_attachments {{
        blend_attach,
        blend_attach,
        blend_attach
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

    array<VkDescriptorSetLayout, 3> draw_desc_layouts {{
        shaders.getLayout(0),
        shaders.getLayout(1),
        shaders.getLayout(2),
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

    VkPipelineVertexInputStateCreateInfo vert_info {};
    VkPipelineInputAssemblyStateCreateInfo input_assembly_info {};
    VkPipelineViewportStateCreateInfo viewport_info {};
    VkPipelineMultisampleStateCreateInfo multisample_info {};
    VkPipelineRasterizationStateCreateInfo raster_info {};

    initCommonDrawPipelineInfo(vert_info, input_assembly_info, 
        viewport_info, multisample_info, raster_info);

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
    depth_info.front.compareOp = VK_COMPARE_OP_ALWAYS;

    // Blend
    VkPipelineColorBlendAttachmentState blend_attach {};
    blend_attach.blendEnable = VK_FALSE;
    blend_attach.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;

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

static Pipeline<1> makeVoxelDrawPipeline(const Device& dev,
    VkPipelineCache pipeline_cache,
    VkRenderPass render_pass,
    VkSampler repeat_sampler,
    VkSampler clamp_sampler,
    uint32_t num_frames)
{
    auto shaders =
        makeVoxelDrawShaders(dev, repeat_sampler, clamp_sampler);

    VkPipelineVertexInputStateCreateInfo vert_info{};
    VkPipelineInputAssemblyStateCreateInfo input_assembly_info{};
    VkPipelineViewportStateCreateInfo viewport_info{};
    VkPipelineMultisampleStateCreateInfo multisample_info{};
    VkPipelineRasterizationStateCreateInfo raster_info{};

    initCommonDrawPipelineInfo(vert_info, input_assembly_info,
        viewport_info, multisample_info, raster_info);

    // Depth/Stencil
    VkPipelineDepthStencilStateCreateInfo depth_info{};
    depth_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_info.depthTestEnable = VK_TRUE;
    depth_info.depthWriteEnable = VK_TRUE;
    depth_info.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
    depth_info.depthBoundsTestEnable = VK_FALSE;
    depth_info.stencilTestEnable = VK_FALSE;
    depth_info.back.compareOp = VK_COMPARE_OP_ALWAYS;

    // Blend
    VkPipelineColorBlendAttachmentState blend_attach{};
    blend_attach.blendEnable = VK_FALSE;
    blend_attach.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    array<VkPipelineColorBlendAttachmentState, 3> blend_attachments {{
            blend_attach,
                blend_attach,
                blend_attach
        }};

    VkPipelineColorBlendStateCreateInfo blend_info{};
    blend_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend_info.logicOpEnable = VK_FALSE;
    blend_info.attachmentCount =
        static_cast<uint32_t>(blend_attachments.size());
    blend_info.pAttachments = blend_attachments.data();

    // Dynamic
    array dyn_enable{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dyn_info{};
    dyn_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn_info.dynamicStateCount = dyn_enable.size();
    dyn_info.pDynamicStates = dyn_enable.data();

    // Push constant
    VkPushConstantRange push_const{
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(DrawPushConst),
    };

    // Layout configuration

    array<VkDescriptorSetLayout, 2> draw_desc_layouts {{
            shaders.getLayout(0),
                shaders.getLayout(1)
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

static Pipeline<1> makeShadowGenPipeline(const Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    VkSampler clamp_sampler,
                                    CountT num_frames)
{
    PipelineShaders shader = makeShadowGenShader(dev, clamp_sampler);

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(shader::ShadowGenPushConst),
    };

    // Layout configuration
    std::array desc_layouts {
        shader.getLayout(0),
    };

    VkPipelineLayoutCreateInfo lighting_layout_info;
    lighting_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lighting_layout_info.pNext = nullptr;
    lighting_layout_info.flags = 0;
    lighting_layout_info.setLayoutCount =
        static_cast<uint32_t>(desc_layouts.size());
    lighting_layout_info.pSetLayouts = desc_layouts.data();
    lighting_layout_info.pushConstantRangeCount = 1;
    lighting_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout lighting_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &lighting_layout_info, nullptr,
                                       &lighting_layout));

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
        "shadowGen",
        nullptr,
    };
    compute_infos[0].layout = lighting_layout;
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
        lighting_layout,
        pipelines,
        std::move(desc_pool),
    };
}

static Pipeline<1> makeDeferredLightingPipeline(const Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    VkSampler clamp_sampler,
                                    CountT num_frames)
{
    PipelineShaders shader = makeDeferredLightingShader(dev, clamp_sampler);

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(DeferredLightingPushConst),
    };

    // Layout configuration
    std::array desc_layouts {
        shader.getLayout(0),
    };

    VkPipelineLayoutCreateInfo lighting_layout_info;
    lighting_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lighting_layout_info.pNext = nullptr;
    lighting_layout_info.flags = 0;
    lighting_layout_info.setLayoutCount =
        static_cast<uint32_t>(desc_layouts.size());
    lighting_layout_info.pSetLayouts = desc_layouts.data();
    lighting_layout_info.pushConstantRangeCount = 1;
    lighting_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout lighting_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &lighting_layout_info, nullptr,
                                       &lighting_layout));

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
        "lighting",
        nullptr,
    };
    compute_infos[0].layout = lighting_layout;
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
        lighting_layout,
        pipelines,
        std::move(desc_pool),
    };
}

static Pipeline<1> makeBlurPipeline(const Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    VkSampler clamp_sampler,
                                    CountT num_frames)
{
    PipelineShaders shader = makeBlurShader(dev, clamp_sampler);

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(shader::BlurPushConst),
    };

    // Layout configuration
    std::array desc_layouts {
        shader.getLayout(0),
    };

    VkPipelineLayoutCreateInfo blur_layout_info;
    blur_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    blur_layout_info.pNext = nullptr;
    blur_layout_info.flags = 0;
    blur_layout_info.setLayoutCount =
        static_cast<uint32_t>(desc_layouts.size());
    blur_layout_info.pSetLayouts = desc_layouts.data();
    blur_layout_info.pushConstantRangeCount = 1;
    blur_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout blur_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &blur_layout_info, nullptr,
                                       &blur_layout));

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
        "blur",
        nullptr,
    };
    compute_infos[0].layout = blur_layout;
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
        blur_layout,
        pipelines,
        std::move(desc_pool),
    };
}

static Backend initializeBackend()
{
    auto get_inst_addr = glfwGetInstanceProcAddress(
        VK_NULL_HANDLE, "vkGetInstanceProcAddr");

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

static std::pair<Framebuffer, Framebuffer> makeFramebuffers(const Device &dev,
                                   MemoryAllocator &alloc,
                                   uint32_t fb_width,
                                   uint32_t fb_height,
                                   VkRenderPass render_pass,
                                   VkRenderPass imgui_render_pass)
{
    auto albedo = alloc.makeColorAttachment(
        fb_width, fb_height, VK_FORMAT_R8G8B8A8_UNORM);
    auto normal = alloc.makeColorAttachment(
        fb_width, fb_height, VK_FORMAT_R16G16B16A16_SFLOAT);
    auto position = alloc.makeColorAttachment(
        fb_width, fb_height, VK_FORMAT_R16G16B16A16_SFLOAT);
    auto depth = alloc.makeDepthAttachment(
        fb_width, fb_height, InternalConfig::depthFormat);

    VkImageViewCreateInfo view_info {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkImageSubresourceRange &view_info_sr = view_info.subresourceRange;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info_sr.baseMipLevel = 0;
    view_info_sr.levelCount = 1;
    view_info_sr.baseArrayLayer = 0;
    view_info_sr.layerCount = 1;

    view_info.image = albedo.image;
    view_info.format = VK_FORMAT_R8G8B8A8_UNORM;

    VkImageView albedo_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &albedo_view));

    view_info.image = normal.image;
    view_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;

    VkImageView normal_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &normal_view));

    view_info.image = position.image;
    view_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;

    VkImageView position_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &position_view));

    view_info.image = depth.image;
    view_info.format = InternalConfig::depthFormat;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    VkImageView depth_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &depth_view));

    array attachment_views {
        albedo_view,
        normal_view,
        position_view,
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

    array imgui_attachment_views {
        albedo_view,
        depth_view,
    };

    VkFramebufferCreateInfo imgui_fb_info;
    imgui_fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    imgui_fb_info.pNext = nullptr;
    imgui_fb_info.flags = 0;
    imgui_fb_info.renderPass = imgui_render_pass;
    imgui_fb_info.attachmentCount = 2;
    imgui_fb_info.pAttachments = imgui_attachment_views.data();
    imgui_fb_info.width = fb_width;
    imgui_fb_info.height = fb_height;
    imgui_fb_info.layers = 1;

    VkFramebuffer imgui_hdl;
    REQ_VK(dev.dt.createFramebuffer(dev.hdl, &imgui_fb_info, nullptr, &imgui_hdl));

    return std::make_pair(
        Framebuffer {
            std::move(albedo),
            std::move(normal),
            std::move(position),
            std::move(depth),
            albedo_view,
            normal_view,
            position_view,
            depth_view,
            hdl 
        },
        Framebuffer {
            std::move(albedo),
            std::move(normal),
            std::move(position),
            std::move(depth),
            albedo_view,
            normal_view,
            position_view,
            depth_view,
            imgui_hdl 
        }
    );
}

static ShadowFramebuffer makeShadowFramebuffer(const Device &dev,
                                   MemoryAllocator &alloc,
                                   uint32_t fb_width,
                                   uint32_t fb_height,
                                   VkRenderPass render_pass)
{
    auto color = alloc.makeColorAttachment(fb_width, fb_height,
        InternalConfig::varianceFormat);
    auto intermediate = alloc.makeColorAttachment(fb_width, fb_height,
        InternalConfig::varianceFormat);
    auto depth = alloc.makeDepthAttachment(
        fb_width, fb_height, InternalConfig::depthFormat);

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
    view_info.format = InternalConfig::depthFormat;

    VkImageView depth_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &depth_view));

    view_info_sr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.image = color.image;
    view_info.format = InternalConfig::varianceFormat;

    VkImageView color_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &color_view));

    view_info.image = intermediate.image;
    view_info.format = InternalConfig::varianceFormat;

    VkImageView intermediate_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &intermediate_view));

    array attachment_views {
        color_view,
        depth_view
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
        std::move(color),
        std::move(intermediate),
        std::move(depth),
        color_view,
        intermediate_view,
        depth_view,
        hdl,
    };
}

static void makeFrame(const Device &dev, MemoryAllocator &alloc,
                      uint32_t fb_width, uint32_t fb_height,
                      uint32_t max_views, uint32_t max_instances,
                      VoxelConfig voxel_config,
                      VkRenderPass render_pass,
                      VkRenderPass imgui_render_pass,
                      VkRenderPass shadow_pass,
                      VkDescriptorSet cull_set,
                      VkDescriptorSet draw_set,
                      VkDescriptorSet lighting_set,
                      VkDescriptorSet shadow_gen_set,
                      VkDescriptorSet shadow_blur_set,
                      VkDescriptorSet voxel_gen_set,
                      VkDescriptorSet voxel_draw_set,
                      Sky &sky,
                      Frame *dst)
{
    auto [fb, imgui_fb] = makeFramebuffers(dev, alloc, fb_width, fb_height, render_pass, imgui_render_pass);
    auto shadow_fb = makeShadowFramebuffer(dev, alloc, InternalConfig::shadowMapSize, InternalConfig::shadowMapSize, shadow_pass);

    VkCommandPool cmd_pool = makeCmdPool(dev, dev.gfxQF);

    int64_t buffer_offsets[7];
    int64_t buffer_sizes[8] = {
        (int64_t)sizeof(PackedViewData) * (max_views + 1),
        (int64_t)sizeof(uint32_t),
        (int64_t)sizeof(PackedInstanceData) * max_instances,
        (int64_t)sizeof(DrawCmd) * max_instances * 10,
        (int64_t)sizeof(DrawData) * max_instances * 10,
        (int64_t)sizeof(DirectionalLight) * InternalConfig::maxLights,
        (int64_t)sizeof(ShadowViewData) * (max_views + 1),
        (int64_t)sizeof(SkyData)
    };

    int64_t num_render_input_bytes = utils::computeBufferOffsets(
        buffer_sizes, buffer_offsets, 256);

    HostBuffer view_staging = alloc.makeStagingBuffer(sizeof(PackedViewData));
    HostBuffer light_staging = alloc.makeStagingBuffer(sizeof(DirectionalLight) * InternalConfig::maxLights);
    // HostBuffer shadow_staging = alloc.makeStagingBuffer(sizeof(ShadowViewData));
    HostBuffer sky_staging = alloc.makeStagingBuffer(sizeof(SkyData));

    LocalBuffer render_input = *alloc.makeLocalBuffer(num_render_input_bytes);

    std::array<VkWriteDescriptorSet, 23> desc_updates;

    VkDescriptorBufferInfo view_info;
    view_info.buffer = render_input.buffer;
    view_info.offset = 0;
    view_info.range = buffer_sizes[0];

    //DescHelper::uniform(desc_updates[0], cull_set, &view_info, 0);
    DescHelper::storage(desc_updates[0], draw_set, &view_info, 0);
    DescHelper::storage(desc_updates[19], shadow_gen_set, &view_info, 1);

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

    VkDescriptorBufferInfo draw_data_info;
    draw_data_info.buffer = render_input.buffer;
    draw_data_info.offset = buffer_offsets[3];
    draw_data_info.range = buffer_sizes[4];

    DescHelper::storage(desc_updates[5], cull_set, &draw_data_info, 4);
    DescHelper::storage(desc_updates[6], draw_set, &draw_data_info, 2);

    VkDescriptorImageInfo gbuffer_albedo_info;
    gbuffer_albedo_info.imageView = fb.colorView;
    gbuffer_albedo_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    gbuffer_albedo_info.sampler = VK_NULL_HANDLE;

    DescHelper::storageImage(desc_updates[7], lighting_set, &gbuffer_albedo_info, 0);

    VkDescriptorImageInfo gbuffer_normal_info;
    gbuffer_normal_info.imageView = fb.normalView;
    gbuffer_normal_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    gbuffer_normal_info.sampler = VK_NULL_HANDLE;

    DescHelper::storageImage(desc_updates[8], lighting_set, &gbuffer_normal_info, 1);

    VkDescriptorImageInfo gbuffer_position_info;
    gbuffer_position_info.imageView = fb.positionView;
    gbuffer_position_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    gbuffer_position_info.sampler = VK_NULL_HANDLE;

    DescHelper::storageImage(desc_updates[9], lighting_set, &gbuffer_position_info, 2);

    VkDescriptorBufferInfo light_data_info;
    light_data_info.buffer = render_input.buffer;
    light_data_info.offset = buffer_offsets[4];
    light_data_info.range = buffer_sizes[5];

    DescHelper::storage(desc_updates[10], lighting_set, &light_data_info, 3);
    DescHelper::storage(desc_updates[20], shadow_gen_set, &light_data_info, 2);

    VkDescriptorImageInfo transmittance_info;
    transmittance_info.imageView = sky.transmittanceView;
    transmittance_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    transmittance_info.sampler = VK_NULL_HANDLE;

    DescHelper::textures(desc_updates[11], lighting_set, &transmittance_info, 1, 4, 0);

    VkDescriptorImageInfo irradiance_info;
    irradiance_info.imageView = sky.irradianceView;
    irradiance_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    irradiance_info.sampler = VK_NULL_HANDLE;

    DescHelper::textures(desc_updates[12], lighting_set, &irradiance_info, 1, 5, 0);

    VkDescriptorImageInfo scattering_info;
    scattering_info.imageView = sky.scatteringView;
    scattering_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    scattering_info.sampler = VK_NULL_HANDLE;

    DescHelper::textures(desc_updates[13], lighting_set, &scattering_info, 1, 6, 0);

    VkDescriptorBufferInfo shadow_view_info;
    shadow_view_info.buffer = render_input.buffer;
    shadow_view_info.offset = buffer_offsets[5];
    shadow_view_info.range = buffer_sizes[6];

    DescHelper::storage(desc_updates[14], draw_set, &shadow_view_info, 3);
    DescHelper::storage(desc_updates[15], lighting_set, &shadow_view_info, 8);
    DescHelper::storage(desc_updates[18], shadow_gen_set, &shadow_view_info, 0);

    VkDescriptorImageInfo shadow_map_info;
    shadow_map_info.imageView = shadow_fb.varianceView;
    // shadow_map_info.imageView = shadow_fb.depthView;
    shadow_map_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    shadow_map_info.sampler = VK_NULL_HANDLE;

    DescHelper::textures(desc_updates[16], lighting_set, &shadow_map_info, 1, 7, 0);

    VkDescriptorBufferInfo sky_info;
    sky_info.buffer = render_input.buffer;
    sky_info.offset = buffer_offsets[6];
    sky_info.range = buffer_sizes[7];

    DescHelper::storage(desc_updates[17], lighting_set, &sky_info, 10);

    VkDescriptorImageInfo blur_input_info;
    blur_input_info.imageView = shadow_fb.varianceView;
    blur_input_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    blur_input_info.sampler = VK_NULL_HANDLE;

    DescHelper::storageImage(desc_updates[21], shadow_blur_set, &blur_input_info, 0);

    VkDescriptorImageInfo blur_intermediate_info;
    blur_intermediate_info.imageView = shadow_fb.intermediateView;
    blur_intermediate_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    blur_intermediate_info.sampler = VK_NULL_HANDLE;

    DescHelper::storageImage(desc_updates[22], shadow_blur_set, &blur_intermediate_info, 1);


    //Voxelizer Changes
    const int32_t num_voxels = voxel_config.xLength * voxel_config.yLength * voxel_config.zLength;
    const int32_t voxels_size = num_voxels > 0 ? sizeof(int32_t) * num_voxels : 4;
    const int32_t vertices_size = num_voxels > 0 ? num_voxels * 32 * 6 * sizeof(float) : 4;
    const int32_t indices_size = num_voxels > 0 ? num_voxels * 6 * 6 * sizeof(int32_t) : 4;

    std::array<VkWriteDescriptorSet, 5> voxel_updates;

    LocalBuffer voxel_vbo = *alloc.makeLocalBuffer(vertices_size);
    LocalBuffer voxel_ibo = *alloc.makeLocalBuffer(indices_size);
    LocalBuffer voxel_data = *alloc.makeLocalBuffer(voxels_size);

    VkDescriptorBufferInfo voxel_vbo_info;
    voxel_vbo_info.buffer = voxel_vbo.buffer;
    voxel_vbo_info.offset = 0;
    voxel_vbo_info.range = vertices_size;

    DescHelper::storage(voxel_updates[0], voxel_gen_set, &voxel_vbo_info, 0);

    VkDescriptorBufferInfo voxelIndexBuffer_info;
    voxelIndexBuffer_info.buffer = voxel_ibo.buffer;
    voxelIndexBuffer_info.offset = 0;
    voxelIndexBuffer_info.range = indices_size;

    DescHelper::storage(voxel_updates[1], voxel_gen_set, &voxelIndexBuffer_info, 1);

    VkDescriptorBufferInfo voxel_info;
    voxel_info.buffer = voxel_data.buffer;
    voxel_info.offset = 0;
    voxel_info.range = voxels_size;

    DescHelper::storage(voxel_updates[2], voxel_gen_set, &voxel_info, 2);

    DescHelper::storage(voxel_updates[3], voxel_draw_set, &view_info, 0);

    VkDescriptorBufferInfo vert_info;
    vert_info.buffer = voxel_vbo.buffer;
    vert_info.offset = 0;
    vert_info.range = vertices_size;
    DescHelper::storage(voxel_updates[4], voxel_draw_set, &vert_info, 1);
    DescHelper::update(dev, voxel_updates.data(), voxel_updates.size());
    //End Voxelizer Changes

    DescHelper::update(dev, desc_updates.data(), desc_updates.size());


    new (dst) Frame {
        std::move(fb),
        std::move(imgui_fb),
        std::move(shadow_fb),
        cmd_pool,
        makeCmdBuffer(dev, cmd_pool),
        makeFence(dev, true),
        makeBinarySemaphore(dev),
        makeBinarySemaphore(dev),
        std::move(view_staging),
        std::move(light_staging),
        std::move(sky_staging),
        std::move(render_input),
        std::move(voxel_vbo),
        std::move(voxel_ibo),
        std::move(voxel_data),
        num_render_input_bytes,
        0,
        sizeof(PackedViewData),
        uint32_t(buffer_offsets[2]),
        uint32_t(buffer_offsets[0]),
        uint32_t(buffer_offsets[1]),
        (uint32_t)buffer_offsets[4],
        (uint32_t)buffer_offsets[5],
        (uint32_t)buffer_offsets[6],
        max_instances * 10,
        cull_set,
        draw_set,
        lighting_set,
        shadow_gen_set,
        shadow_blur_set,
        voxel_gen_set,
        voxel_draw_set
    };
}

static array<VkClearValue, 4> makeClearValues()
{
    VkClearValue color_clear;
    color_clear.color = {{0.f, 0.f, 0.f, 0.f}};

    VkClearValue depth_clear;
    depth_clear.depthStencil = {0.f, 0};

    return {
        color_clear,
        color_clear,
        color_clear,
        depth_clear,
    };
}

static array<VkClearValue, 2> makeShadowClearValues()
{
    VkClearValue color_clear = {};
    color_clear.color = {{0.f, 0.f, 0.f, 0.f}};

    VkClearValue depth_clear = {};
    depth_clear.depthStencil = {0.f, 0};

    return {
        depth_clear,
    };
}

static array<VkClearValue, 2> makeImguiClearValues()
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
    io.Fonts->AddFontFromFileTTF(font_path.string().c_str(), scaled_font_size);

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
}//zoe lynch

static EngineInterop setupEngineInterop(Device &dev,
                                        MemoryAllocator &alloc,
                                        bool gpu_input,
                                        uint32_t gpu_id,
                                        uint32_t num_worlds,
                                        uint32_t max_views_per_world,
                                        uint32_t max_instances_per_world,
                                        uint32_t render_width,
                                        uint32_t render_height,
                                        VoxelConfig voxel_config)
{
    int64_t render_input_buffer_offsets[1];
    int64_t render_input_buffer_sizes[2] = {
        (int64_t)sizeof(shader::PackedInstanceData) *
            num_worlds * max_instances_per_world,
        (int64_t)sizeof(shader::PackedViewData) *
            num_worlds * max_views_per_world,
    };

    int64_t num_render_input_bytes = utils::computeBufferOffsets(
        render_input_buffer_sizes, render_input_buffer_offsets, 256);

    auto render_input_cpu = Optional<render::vk::HostBuffer>::none();
#ifdef MADRONA_CUDA_SUPPORT
    auto render_input_gpu = Optional<render::vk::DedicatedBuffer>::none();
    auto render_input_cuda = Optional<render::vk::CudaImportedBuffer>::none();
#endif

    VkBuffer render_input_hdl;
    char *render_input_base;
    if (!gpu_input) {
        render_input_cpu = alloc.makeStagingBuffer(num_render_input_bytes);
        render_input_hdl = render_input_cpu->buffer;
        render_input_base = (char *)render_input_cpu->ptr;
    } else {
#ifdef MADRONA_CUDA_SUPPORT
        render_input_gpu = alloc.makeDedicatedBuffer(
            num_render_input_bytes, false, true);

        render_input_cuda.emplace(dev, gpu_id, render_input_gpu->mem,
                                  num_render_input_bytes);

        render_input_hdl = render_input_gpu->buf.buffer;
        render_input_base = (char *)render_input_cuda->getDevicePointer();
#else
        (void)dev;
        (void)gpu_id;
        render_input_hdl = VK_NULL_HANDLE;
        render_input_base = nullptr;
#endif
    }

    InstanceData *instance_base =
        (InstanceData *)render_input_base;

    PerspectiveCameraData *view_base = 
        (PerspectiveCameraData *)(render_input_base + render_input_buffer_offsets[0]);

    InstanceData **world_instances_setup;
    PerspectiveCameraData **world_views_setup;

#ifdef MADRONA_CUDA_SUPPORT
    auto setup_alloc = gpu_input ? cu::allocStaging : malloc;
#else
    auto setup_alloc = malloc;
#endif

    uint64_t num_world_inst_setup_bytes =
        sizeof(InstanceData *) * num_worlds;
    uint64_t num_world_view_setup_bytes =
        sizeof(PerspectiveCameraData *) * num_worlds;
   
    world_instances_setup = (InstanceData **)setup_alloc(
        num_world_inst_setup_bytes);
    
    world_views_setup = (PerspectiveCameraData **)setup_alloc(
        num_world_view_setup_bytes);

    for (CountT i = 0; i < (CountT)num_worlds; i++) {
        world_instances_setup[i] = instance_base + i * max_instances_per_world;
        world_views_setup[i] = view_base + i * max_views_per_world;
    }

    InstanceData **world_instances;
    PerspectiveCameraData **world_views;

    if (!gpu_input) {
        world_instances = world_instances_setup;
        world_views = world_views_setup;
    } else {
#ifdef MADRONA_CUDA_SUPPORT
        world_instances = (InstanceData **)cu::allocGPU(
            num_world_inst_setup_bytes);
        world_views = (PerspectiveCameraData **)cu::allocGPU(
            num_world_view_setup_bytes);

        cudaMemcpy(world_instances, world_instances_setup,
            num_world_inst_setup_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(world_views, world_views_setup,
            num_world_view_setup_bytes, cudaMemcpyHostToDevice);
#else
        world_instances = nullptr;
        world_views = nullptr;
#endif
    }

    uint32_t *num_views, *num_instances;

    if (!gpu_input) {
        uint32_t *counts = (uint32_t *)malloc(
            sizeof(uint32_t) * 2 * num_worlds);

        num_views = counts;
        num_instances = counts + num_worlds;
    } else {
#ifdef MADRONA_CUDA_SUPPORT
        uint32_t *counts = (uint32_t *)cu::allocReadback(
            sizeof(uint32_t) * 2 * num_worlds);

        num_views = counts;
        num_instances = counts + num_worlds;
#else
        num_views = nullptr;
        num_instances = nullptr;
#endif
    }

    const uint32_t num_voxels = voxel_config.xLength
        * voxel_config.yLength * voxel_config.zLength;
    const uint32_t staging_size = num_voxels > 0 ? num_voxels * sizeof(int32_t) : 4;

    auto voxel_cpu = Optional<HostBuffer>::none();
    VkBuffer voxel_buffer_hdl;
    uint32_t *voxel_buffer_ptr;

#ifdef MADRONA_CUDA_SUPPORT
    auto voxel_gpu = Optional<render::vk::DedicatedBuffer>::none();
    auto voxel_cuda = Optional<render::vk::CudaImportedBuffer>::none();
#endif

    if (!gpu_input) {
        voxel_cpu = alloc.makeStagingBuffer(staging_size);
        voxel_buffer_ptr = num_voxels ? (uint32_t *)voxel_cpu->ptr : nullptr;
        voxel_buffer_hdl = voxel_cpu->buffer;
    } else {
#ifdef MADRONA_CUDA_SUPPORT
        voxel_gpu = alloc.makeDedicatedBuffer(
            staging_size, false, true);

        voxel_cuda.emplace(dev, gpu_id, voxel_gpu->mem,
            staging_size);

        voxel_buffer_hdl = voxel_gpu->buf.buffer;
        voxel_buffer_ptr = num_voxels ?
                (uint32_t *)voxel_cuda->getDevicePointer() : nullptr;
#else
        voxel_buffer_ptr = nullptr;
#endif
    }

    VizECSBridge bridge {
        .views = world_views,
        .numViews = num_views,
        .instances = world_instances,
        .numInstances = num_instances,
        .renderWidth = (int32_t)render_width,
        .renderHeight = (int32_t)render_height,
        .episodeDone = nullptr,
        .voxels = voxel_buffer_ptr,
    };

    const VizECSBridge *gpu_bridge;
    if (!gpu_input) {
        gpu_bridge = nullptr;
    } else {
#ifdef MADRONA_CUDA_SUPPORT
        gpu_bridge = (const VizECSBridge *)cu::allocGPU(sizeof(VizECSBridge));
        cudaMemcpy((void *)gpu_bridge, &bridge, sizeof(VizECSBridge),
                   cudaMemcpyHostToDevice);
#else
        gpu_bridge = nullptr;
#endif
    }

    return EngineInterop {
        std::move(render_input_cpu),
#ifdef MADRONA_CUDA_SUPPORT
        std::move(render_input_gpu),
        std::move(render_input_cuda),
#endif
        render_input_hdl,
        bridge,
        gpu_bridge,
        uint32_t(render_input_buffer_offsets[0]),
        max_views_per_world,
        max_instances_per_world,
        std::move(voxel_cpu),
#ifdef MADRONA_CUDA_SUPPORT
        std::move(voxel_gpu),
        std::move(voxel_cuda),
#endif
        voxel_buffer_hdl,
    };
}

static size_t fileLength(std::fstream &f)
{
    f.seekg(0, f.end);
    size_t size = f.tellg();
    f.seekg(0, f.beg);

    return size;
}

inline constexpr size_t TRANSMITTANCE_WIDTH = 256;
inline constexpr size_t TRANSMITTANCE_HEIGHT = 64;
inline constexpr size_t SCATTERING_TEXTURE_R_SIZE = 32;
inline constexpr size_t SCATTERING_TEXTURE_MU_SIZE = 128;
inline constexpr size_t SCATTERING_TEXTURE_MU_S_SIZE = 32;
inline constexpr size_t SCATTERING_TEXTURE_NU_SIZE = 8;
inline constexpr size_t SCATTERING_TEXTURE_WIDTH =
    SCATTERING_TEXTURE_NU_SIZE * SCATTERING_TEXTURE_MU_S_SIZE;
inline constexpr size_t SCATTERING_TEXTURE_HEIGHT = SCATTERING_TEXTURE_MU_SIZE;
inline constexpr size_t SCATTERING_TEXTURE_DEPTH = SCATTERING_TEXTURE_R_SIZE;
inline constexpr size_t IRRADIANCE_TEXTURE_WIDTH = 64;
inline constexpr size_t IRRADIANCE_TEXTURE_HEIGHT = 16;
inline constexpr size_t SHADOW_OFFSET_OUTER = 32;
inline constexpr size_t SHADOW_OFFSET_FILTER_SIZE = 8;

static Sky loadSky(const vk::Device &dev, MemoryAllocator &alloc, VkQueue queue)
{
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(VIEWER_DATA_DIR)) /
        "sky";

    auto [transmittance, transmittance_reqs] = alloc.makeTexture2D(
        TRANSMITTANCE_WIDTH, TRANSMITTANCE_HEIGHT, 1, InternalConfig::skyFormatHighp);
    auto [irradiance, irradiance_reqs] = alloc.makeTexture2D(
        IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT, 1, InternalConfig::skyFormatHighp);
    auto [mie, mie_reqs] = alloc.makeTexture3D(
        SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH, 1, InternalConfig::skyFormatHalfp);
    auto [scattering, scattering_reqs] = alloc.makeTexture3D(
        SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH, 1, InternalConfig::skyFormatHalfp);

    HostBuffer irradiance_hb_staging = alloc.makeStagingBuffer(irradiance_reqs.size);
    HostBuffer mie_hb_staging = alloc.makeStagingBuffer(mie_reqs.size);
    HostBuffer scattering_hb_staging = alloc.makeStagingBuffer(scattering_reqs.size);
    HostBuffer transmittance_hb_staging = alloc.makeStagingBuffer(transmittance_reqs.size);

    std::fstream irradiance_stream(shader_dir / "irradiance.cache", std::fstream::binary | std::fstream::in);
    size_t irradiance_size = fileLength(irradiance_stream);
    irradiance_stream.read((char *)irradiance_hb_staging.ptr, irradiance_size);
    irradiance_hb_staging.flush(dev);

    std::fstream mie_stream(shader_dir / "mie.cache", std::fstream::binary | std::fstream::in);
    size_t mie_size = fileLength(mie_stream);
    mie_stream.read((char *)mie_hb_staging.ptr, mie_size);
    mie_hb_staging.flush(dev);

    std::fstream scattering_stream(shader_dir / "scattering.cache", std::fstream::binary | std::fstream::in);
    size_t scattering_size = fileLength(scattering_stream);
    scattering_stream.read((char *)scattering_hb_staging.ptr, scattering_size);
    scattering_hb_staging.flush(dev);

    std::fstream transmittance_stream(shader_dir / "transmittance.cache", std::fstream::binary | std::fstream::in);
    size_t transmittance_size = fileLength(transmittance_stream);
    transmittance_stream.read((char *)transmittance_hb_staging.ptr, transmittance_size);
    transmittance_hb_staging.flush(dev);

#if 1
    assert(transmittance_size == transmittance_reqs.size && irradiance_size == irradiance_reqs.size &&
           scattering_size == scattering_reqs.size && mie_size == mie_reqs.size);
#endif

    std::optional<VkDeviceMemory> transmittance_backing = alloc.alloc(transmittance_reqs.size);
    std::optional<VkDeviceMemory> irradiance_backing = alloc.alloc(irradiance_reqs.size);
    std::optional<VkDeviceMemory> scattering_backing = alloc.alloc(scattering_reqs.size);
    std::optional<VkDeviceMemory> mie_backing = alloc.alloc(mie_reqs.size);

    assert(transmittance_backing.has_value() && irradiance_backing.has_value() &&
        scattering_backing.has_value() && mie_backing.has_value());

    dev.dt.bindImageMemory(dev.hdl, transmittance.image, transmittance_backing.value(), 0);
    dev.dt.bindImageMemory(dev.hdl, irradiance.image, irradiance_backing.value(), 0);
    dev.dt.bindImageMemory(dev.hdl, scattering.image, scattering_backing.value(), 0);
    dev.dt.bindImageMemory(dev.hdl, mie.image, mie_backing.value(), 0);

    VkCommandPool tmp_pool = makeCmdPool(dev, dev.gfxQF);
    VkCommandBuffer cmdbuf = makeCmdBuffer(dev, tmp_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    VkCommandBufferBeginInfo begin_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        nullptr
    };

    dev.dt.beginCommandBuffer(cmdbuf, &begin_info);
    {
        array<VkImageMemoryBarrier, 4> copy_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                0,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                transmittance.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                0,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                irradiance.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                0,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                mie.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                0,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                scattering.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            }
        }};

        dev.dt.cmdPipelineBarrier(cmdbuf,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr, 0, nullptr,
                copy_prepare.size(), copy_prepare.data());

#if 1
        VkBufferImageCopy copy = {};
        copy.bufferOffset = 0;
        copy.bufferRowLength = 0;
        copy.bufferImageHeight = 0;
        copy.imageExtent.width = TRANSMITTANCE_WIDTH;
        copy.imageExtent.height = TRANSMITTANCE_HEIGHT;
        copy.imageExtent.depth = 1;
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.mipLevel = 0;
        copy.imageSubresource.baseArrayLayer = 0;
        copy.imageSubresource.layerCount = 1;

        dev.dt.cmdCopyBufferToImage(cmdbuf, transmittance_hb_staging.buffer,
            transmittance.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        copy.imageExtent.width = IRRADIANCE_TEXTURE_WIDTH;
        copy.imageExtent.height = IRRADIANCE_TEXTURE_HEIGHT;
        dev.dt.cmdCopyBufferToImage(cmdbuf, irradiance_hb_staging.buffer,
            irradiance.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        copy.imageExtent.width = SCATTERING_TEXTURE_WIDTH; 
        copy.imageExtent.height = SCATTERING_TEXTURE_HEIGHT;
        copy.imageExtent.depth = SCATTERING_TEXTURE_DEPTH;
        dev.dt.cmdCopyBufferToImage(cmdbuf, mie_hb_staging.buffer,
            mie.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        dev.dt.cmdCopyBufferToImage(cmdbuf, scattering_hb_staging.buffer,
            scattering.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

#endif

        array<VkImageMemoryBarrier, 4> finish_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                transmittance.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                irradiance.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                mie.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                scattering.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            }
        }};

        dev.dt.cmdPipelineBarrier(cmdbuf,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr, 0, nullptr,
                finish_prepare.size(), finish_prepare.data());
    }
    dev.dt.endCommandBuffer(cmdbuf);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmdbuf;
    submit_info.signalSemaphoreCount = 0;
    submit_info.waitSemaphoreCount = 0;

    dev.dt.queueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);

    dev.dt.deviceWaitIdle(dev.hdl);
    dev.dt.freeCommandBuffers(dev.hdl, tmp_pool, 1, &cmdbuf);
    dev.dt.destroyCommandPool(dev.hdl, tmp_pool, nullptr);

    VkImageViewCreateInfo view_info {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkImageSubresourceRange &view_info_sr = view_info.subresourceRange;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info_sr.baseMipLevel = 0;
    view_info_sr.levelCount = 1;
    view_info_sr.baseArrayLayer = 0;
    view_info_sr.layerCount = 1;

    VkImageView transmittance_view;
    view_info.image = transmittance.image;
    view_info.format = InternalConfig::skyFormatHighp;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &transmittance_view));

    VkImageView irradiance_view;
    view_info.image = irradiance.image;
    view_info.format = InternalConfig::skyFormatHighp;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &irradiance_view));

    VkImageView mie_view;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
    view_info.image = mie.image;
    view_info.format = InternalConfig::skyFormatHalfp;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &mie_view));

    VkImageView scattering_view;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
    view_info.image = scattering.image;
    view_info.format = InternalConfig::skyFormatHalfp;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &scattering_view));

    return Sky{
        std::move(transmittance),
        std::move(scattering),
        std::move(mie),
        std::move(irradiance),
        transmittance_view,
        scattering_view,
        mie_view,
        irradiance_view,
        transmittance_backing.value(),
        scattering_backing.value(),
        mie_backing.value(),
        irradiance_backing.value(),
        math::Vector3{ -0.4f, -0.4f, -1.0f },
        math::Vector3{2.0f, 2.0f, 2.0f},
        math::Vector3{0.0046750340586467079f, 0.99998907220740285f, 0.0f},
        20.0f
    };
}
//zoe lynch
void Renderer::createTimeBuffer() {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeof(float); // Assuming you are storing a single float for time
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (dev.dt.createBuffer(dev.hdl, &bufferInfo, nullptr, &timeBuffer_) != VK_SUCCESS) {//.device(), &bufferInfo, nullptr, &timeBuffer_) != VK_SUCCESS) {
        std::cout << "failed to create time buffer!" << std::endl;
        //throw std::runtime_error("failed to create time buffer!");
    }

    VkMemoryRequirements memRequirements;
    dev.dt.getBufferMemoryRequirements(dev.hdl, timeBuffer_, &memRequirements);//.device(), timeBuffer_, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    //allocInfo.memoryTypeIndex = findMemoryType(dev.hdl, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    allocInfo.memoryTypeIndex = findMemoryType(instanceDispatch_,dev.phy, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);


    if (dev.dt.allocateMemory(dev.hdl, &allocInfo, nullptr, &timeBufferMemory_) != VK_SUCCESS) {
        std::cout << "failed to allocate time buffer memory!" << std::endl;
        //throw std::runtime_error("failed to allocate time buffer memory!");
    }

    dev.dt.bindBufferMemory(dev.hdl, timeBuffer_, timeBufferMemory_, 0);
}

void Renderer::createTimeBufferDescriptorSet() {
    // Create the descriptor set layout
    VkDescriptorSetLayoutBinding timeLayoutBinding{};
    timeLayoutBinding.binding = 0;
    timeLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    timeLayoutBinding.descriptorCount = 1;
    timeLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &timeLayoutBinding;

    if (dev.dt.createDescriptorSetLayout(dev.hdl, &layoutInfo, nullptr, &timeBufferLayout_) != VK_SUCCESS) {
        std::cout <<"failed to create time buffer descriptor set layout!" << std::endl;
    }
    
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = 1; // Adjust based on how many descriptors you need

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1; // The maximum number of descriptor sets that can be allocated from the pool
    VkDescriptorPool descriptorPool;
    if (dev.dt.createDescriptorPool(dev.hdl, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        std::cout <<"failed to create descriptor pool!" << std::endl;
    }


    // Allocate the descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &timeBufferLayout_;

    if (dev.dt.allocateDescriptorSets(dev.hdl, &allocInfo, &timeBufferDescriptorSet_) != VK_SUCCESS) {
        std::cout << "failed to allocate time buffer descriptor set!"<< std::endl;
    }

    // Update the descriptor set
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = timeBuffer_;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(float);

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = timeBufferDescriptorSet_;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &bufferInfo;

    dev.dt.updateDescriptorSets(dev.hdl, 1, &descriptorWrite, 0, nullptr);
}


/*
void createBuffer(madrona::render::vk::Device& dev, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory, ImGUIVkLookupData* user_data) {
    std::cout << "Entering createBuffer" << std::endl;
    
    // Create buffer
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (dev.dt.createBuffer(dev.hdl, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        std::cout << "Failed to create buffer!" << std::endl;
    }
    std::cout << "Buffer created successfully" << std::endl;

    // Get memory requirements
    VkMemoryRequirements memRequirements;
    dev.dt.getBufferMemoryRequirements(dev.hdl, buffer, &memRequirements);

    // Cast the user_data to the appropriate type
    if (!user_data) {
        std::cout << "user_data is nullptr!" << std::endl;
    }

    InstanceDispatch instanceDispatch(user_data->inst, user_data->getInstAddr, true);

    // Allocate memory
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(instanceDispatch, dev.phy, memRequirements.memoryTypeBits, properties);

    if (dev.dt.allocateMemory(dev.hdl, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        std::cout << "Failed to allocate buffer memory!" << std::endl;
    }
    std::cout << "Memory allocated successfully" << std::endl;

    // Bind buffer memory
    if (dev.dt.bindBufferMemory(dev.hdl, buffer, bufferMemory, 0) != VK_SUCCESS) {
        std::cout << "Failed to bind buffer memory!" << std::endl;
    }
    std::cout << "Exiting createBuffer" << std::endl;
}*/




Renderer::Renderer(uint32_t gpu_id,
                   uint32_t img_width,
                   uint32_t img_height,
                   uint32_t num_worlds,
                   uint32_t max_views_per_world,
                   uint32_t max_instances_per_world,
                   bool gpu_input,
                   VoxelConfig voxel_config)
    : loader_lib_(PresentationState::init()),
      backend(initializeBackend()),
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
      present_wrapper_(render_queue_, false),
      fb_width_(img_width),
      fb_height_(img_height),
      fb_clear_(makeClearValues()),
      fb_shadow_clear_(makeShadowClearValues()),
      fb_imgui_clear_(makeImguiClearValues()),
      present_(backend, dev, window,
               InternalConfig::numFrames, true),
      pipeline_cache_(getPipelineCache(dev)),
      repeat_sampler_(
          makeImmutableSampler(dev, VK_SAMPLER_ADDRESS_MODE_REPEAT)),
      clamp_sampler_(
          makeImmutableSampler(dev, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)),
      render_pass_(makeRenderPass(dev, VK_FORMAT_R8G8B8A8_UNORM,
                                  InternalConfig::gbufferFormat, InternalConfig::gbufferFormat,
                                  InternalConfig::depthFormat)),
      shadow_pass_(makeShadowRenderPass(dev, InternalConfig::varianceFormat, InternalConfig::depthFormat)),
      imgui_render_state_(imguiInit(window.platformWindow, dev, backend,
                                 render_queue_, pipeline_cache_,
                                 VK_FORMAT_R8G8B8A8_UNORM,
                                 InternalConfig::depthFormat)),
      instance_cull_(makeCullPipeline(dev, pipeline_cache_,
                                      InternalConfig::numFrames)),
      object_draw_(makeDrawPipeline(dev, pipeline_cache_, render_pass_,
                                    repeat_sampler_, clamp_sampler_,
                                    InternalConfig::numFrames)),
      object_shadow_draw_(makeShadowDrawPipeline(dev, pipeline_cache_, shadow_pass_,
                                    repeat_sampler_, clamp_sampler_,
                                    InternalConfig::numFrames)),
      deferred_lighting_(makeDeferredLightingPipeline(dev, pipeline_cache_,
                                      clamp_sampler_,
                                      InternalConfig::numFrames)),
      shadow_gen_(makeShadowGenPipeline(dev, pipeline_cache_, clamp_sampler_, InternalConfig::numFrames)),
      blur_(makeBlurPipeline(dev, pipeline_cache_, clamp_sampler_, InternalConfig::numFrames)),
      //Voxelizer Changes
      voxel_mesh_gen_(makeVoxelMeshGenPipeline(dev,pipeline_cache_,InternalConfig::numFrames)),
      voxel_draw_(makeVoxelDrawPipeline(dev, pipeline_cache_, render_pass_,
          repeat_sampler_, clamp_sampler_,
          InternalConfig::numFrames)),

      asset_desc_pool_cull_(dev, instance_cull_.shaders, 1, 1),
      asset_desc_pool_draw_(dev, object_draw_.shaders, 1, 1),
      asset_desc_pool_mat_tx_(dev, object_draw_.shaders, 2, 1),
      asset_set_cull_(asset_desc_pool_cull_.makeSet()),
      asset_set_draw_(asset_desc_pool_draw_.makeSet()),
      asset_set_mat_tex_(asset_desc_pool_mat_tx_.makeSet()),
      load_cmd_pool_(makeCmdPool(dev, dev.gfxQF)),
      load_cmd_(makeCmdBuffer(dev, load_cmd_pool_)),
      load_fence_(makeFence(dev)),
      engine_interop_(setupEngineInterop(
          dev, alloc, gpu_input, gpu_id, num_worlds,
          max_views_per_world, max_instances_per_world,
          fb_width_, fb_height_, voxel_config)),
      lights_(InternalConfig::maxLights),
      cur_frame_(0),
      frames_(InternalConfig::numFrames),
      loaded_assets_(0),
      sky_(loadSky(dev, alloc, render_queue_)),
      material_textures_(0),
      voxel_config_(voxel_config),
      screenshot_buffer_(),
    instanceDispatch_(backend.hdl, backend.dt.getInstanceProcAddr, true)
{
    for (int i = 0; i < (int)frames_.size(); i++) {
        makeFrame(dev, alloc, fb_width_, fb_height_,
                  max_views_per_world, max_instances_per_world,
                  voxel_config,
                  render_pass_,
                  imgui_render_state_.renderPass,
                  shadow_pass_,
                  instance_cull_.descPool.makeSet(),
                  object_draw_.descPool.makeSet(),
                  deferred_lighting_.descPool.makeSet(),
                  shadow_gen_.descPool.makeSet(),
                  blur_.descPool.makeSet(),
                  voxel_mesh_gen_.descPool.makeSet(),
                  voxel_draw_.descPool.makeSet(),
                  sky_,
                  &frames_[i]);
    }
    
    //zoelynch 1
    /*ImGUIVkLookupData lookup_data1 {
        dev.dt.getDeviceProcAddr,
        dev.hdl,
        backend.dt.getInstanceProcAddr,
        backend.hdl,
    };
    createBuffer(dev, sizeof(float),
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 timeBuffer_, timeBufferMemory_, &lookup_data1); // Pass user_data as the last argument*/
    createTimeBuffer();
    createTimeBufferDescriptorSet();

    screenshot_buffer_ = std::make_unique<render::vk::HostBuffer>(alloc.makeStagingBuffer(frames_[0].fb.colorAttachment.reqs.size));
}

Renderer::~Renderer()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    loaded_assets_.clear();
    //dev.dt.destroyDescriptorPool(dev.hdl, descriptorPool, nullptr);


    for (Frame &f : frames_) {
        dev.dt.destroySemaphore(dev.hdl, f.swapchainReady, nullptr);
        dev.dt.destroySemaphore(dev.hdl, f.renderFinished, nullptr);

        dev.dt.destroyFence(dev.hdl, f.cpuFinished, nullptr);
        dev.dt.destroyCommandPool(dev.hdl, f.cmdPool, nullptr);

        dev.dt.destroyFramebuffer(dev.hdl, f.fb.hdl, nullptr);
        dev.dt.destroyFramebuffer(dev.hdl, f.imguiFBO.hdl, nullptr);
        dev.dt.destroyFramebuffer(dev.hdl, f.shadowFB.hdl, nullptr);

        dev.dt.destroyImageView(dev.hdl, f.fb.colorView, nullptr);
        dev.dt.destroyImageView(dev.hdl, f.fb.normalView, nullptr);
        dev.dt.destroyImageView(dev.hdl, f.fb.positionView, nullptr);
        dev.dt.destroyImageView(dev.hdl, f.fb.depthView, nullptr);

#if 0
        dev.dt.destroyImage(dev.hdl, f.fb.colorAttachment.image, nullptr);
        dev.dt.destroyImage(dev.hdl, f.fb.normalAttachment.image, nullptr);
        dev.dt.destroyImage(dev.hdl, f.fb.positionAttachment.image, nullptr);
        dev.dt.destroyImage(dev.hdl, f.fb.depthAttachment.image, nullptr);
        dev.dt.destroyImage(dev.hdl, f.shadowFB.depthAttachment.image, nullptr);
#endif

        dev.dt.destroyImageView(dev.hdl, f.shadowFB.varianceView, nullptr);
        dev.dt.destroyImageView(dev.hdl, f.shadowFB.depthView, nullptr);

        dev.dt.destroyImageView(dev.hdl, f.shadowFB.intermediateView, nullptr);
    }

    for (auto &tx : material_textures_) {
        dev.dt.destroyImageView(dev.hdl, tx.view, nullptr);
        dev.dt.destroyImage(dev.hdl, tx.image.image, nullptr);
        dev.dt.freeMemory(dev.hdl, tx.backing, nullptr);
    }

    dev.dt.destroyImageView(dev.hdl, sky_.transmittanceView, nullptr);
    dev.dt.destroyImageView(dev.hdl, sky_.irradianceView, nullptr);
    dev.dt.destroyImageView(dev.hdl, sky_.mieView, nullptr);
    dev.dt.destroyImageView(dev.hdl, sky_.scatteringView, nullptr);

    dev.dt.freeMemory(dev.hdl, sky_.transmittanceBacking, nullptr);
    dev.dt.freeMemory(dev.hdl, sky_.irradianceBacking, nullptr);
    dev.dt.freeMemory(dev.hdl, sky_.mieBacking, nullptr);
    dev.dt.freeMemory(dev.hdl, sky_.scatteringBacking, nullptr);

    dev.dt.destroyImage(dev.hdl, sky_.transmittance.image, nullptr);
    dev.dt.destroyImage(dev.hdl, sky_.irradiance.image, nullptr);
    dev.dt.destroyImage(dev.hdl, sky_.singleMieScattering.image, nullptr);
    dev.dt.destroyImage(dev.hdl, sky_.scattering.image, nullptr);

    dev.dt.destroyFence(dev.hdl, load_fence_, nullptr);
    dev.dt.destroyCommandPool(dev.hdl, load_cmd_pool_, nullptr);

    dev.dt.destroyPipeline(dev.hdl, shadow_gen_.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, shadow_gen_.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, object_draw_.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, object_draw_.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, instance_cull_.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, instance_cull_.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, object_shadow_draw_.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, object_shadow_draw_.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, deferred_lighting_.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, deferred_lighting_.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, blur_.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, blur_.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, voxel_mesh_gen_.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, voxel_mesh_gen_.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, voxel_draw_.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, voxel_draw_.layout, nullptr);

    dev.dt.destroyRenderPass(dev.hdl, imgui_render_state_.renderPass, nullptr);
    dev.dt.destroyDescriptorPool(
        dev.hdl, imgui_render_state_.descPool, nullptr);

    dev.dt.destroyRenderPass(dev.hdl, render_pass_, nullptr);
    dev.dt.destroyRenderPass(dev.hdl, shadow_pass_, nullptr);

    dev.dt.destroySampler(dev.hdl, clamp_sampler_, nullptr);
    dev.dt.destroySampler(dev.hdl, repeat_sampler_, nullptr);

    dev.dt.destroyPipelineCache(dev.hdl, pipeline_cache_, nullptr);

    present_.destroy(dev);

    backend.dt.destroySurfaceKHR(backend.hdl, window.surface, nullptr);
    glfwDestroyWindow(window.platformWindow);
}

static DynArray<MaterialTexture> 
loadTextures(const vk::Device &dev, MemoryAllocator &alloc, VkQueue queue,
             Span<const imp::SourceTexture> textures)
{
    DynArray<HostBuffer> host_buffers(0);
    DynArray<MaterialTexture> dst_textures(0);

    VkCommandPool tmp_pool = makeCmdPool(dev, dev.gfxQF);

    VkCommandBuffer cmdbuf = makeCmdBuffer(dev, tmp_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    VkCommandBufferBeginInfo begin_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        nullptr
    };

    dev.dt.beginCommandBuffer(cmdbuf, &begin_info);

    for (const imp::SourceTexture &tx : textures)
    {
        const char *filename = tx.path;
        int width, height, components;
        void *pixels = stbi_load(filename, &width, &height, &components, STBI_rgb_alpha);

        auto [texture, texture_reqs] = alloc.makeTexture2D(
                width, height, 1, VK_FORMAT_R8G8B8A8_SRGB);

        HostBuffer texture_hb_staging = alloc.makeStagingBuffer(texture_reqs.size);
        memcpy(texture_hb_staging.ptr, pixels, width * height * 4 * sizeof(char));
        texture_hb_staging.flush(dev);
        stbi_image_free(pixels);

        std::optional<VkDeviceMemory> texture_backing = alloc.alloc(texture_reqs.size);

        assert(texture_backing.has_value());

        dev.dt.bindImageMemory(dev.hdl, texture.image, texture_backing.value(), 0);

        VkImageMemoryBarrier copy_prepare {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            0,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            texture.image,
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            },
        };

        dev.dt.cmdPipelineBarrier(cmdbuf,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr, 0, nullptr,
            1, &copy_prepare);

        VkBufferImageCopy copy = {};
        copy.bufferOffset = 0;
        copy.bufferRowLength = 0;
        copy.bufferImageHeight = 0;
        copy.imageExtent.width = width;
        copy.imageExtent.height = height;
        copy.imageExtent.depth = 1;
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.mipLevel = 0;
        copy.imageSubresource.baseArrayLayer = 0;
        copy.imageSubresource.layerCount = 1;

        dev.dt.cmdCopyBufferToImage(cmdbuf, texture_hb_staging.buffer,
            texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        VkImageMemoryBarrier finish_prepare {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            texture.image,
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            },
        };
        

        dev.dt.cmdPipelineBarrier(cmdbuf,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr, 0, nullptr,
            1, &finish_prepare);

        VkImageViewCreateInfo view_info {};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        VkImageSubresourceRange &view_info_sr = view_info.subresourceRange;
        view_info_sr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info_sr.baseMipLevel = 0;
        view_info_sr.levelCount = 1;
        view_info_sr.baseArrayLayer = 0;
        view_info_sr.layerCount = 1;

        VkImageView view;
        view_info.image = texture.image;
        view_info.format = VK_FORMAT_R8G8B8A8_SRGB;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &view));

        host_buffers.push_back(std::move(texture_hb_staging));

        dst_textures.emplace_back(std::move(texture), view, texture_backing.value());
    }

    dev.dt.endCommandBuffer(cmdbuf);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmdbuf;
    submit_info.signalSemaphoreCount = 0;
    submit_info.waitSemaphoreCount = 0;

    dev.dt.queueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);

    dev.dt.deviceWaitIdle(dev.hdl);
    dev.dt.freeCommandBuffers(dev.hdl, tmp_pool, 1, &cmdbuf);
    dev.dt.destroyCommandPool(dev.hdl, tmp_pool, nullptr);

    return dst_textures;
}

CountT Renderer::loadObjects(Span<const imp::SourceObject> src_objs,
                             Span<const imp::SourceMaterial> src_mats,
                             Span<const imp::SourceTexture> textures)
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
        materials_ptr[mat_idx].color = mat.color;
        materials_ptr[mat_idx].roughness = mat.roughness;
        materials_ptr[mat_idx].metalness = mat.metalness;
        materials_ptr[mat_idx++].textureIdx = mat.textureIdx;
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

    // std::array<VkWriteDescriptorSet, 5> desc_updates;
    DynArray<VkWriteDescriptorSet> desc_updates(4 + (material_textures_.size() > 0 ? 1 : 0));

    VkDescriptorBufferInfo obj_info;
    obj_info.buffer = asset_buffer.buffer;
    obj_info.offset = 0;
    obj_info.range = buffer_sizes[0];

    desc_updates.push_back({});
    DescHelper::storage(desc_updates[0], asset_set_cull_, &obj_info, 0);

    VkDescriptorBufferInfo mesh_info;
    mesh_info.buffer = asset_buffer.buffer;
    mesh_info.offset = buffer_offsets[0];
    mesh_info.range = buffer_sizes[1];

    desc_updates.push_back({});
    DescHelper::storage(desc_updates[1], asset_set_cull_, &mesh_info, 1);

    VkDescriptorBufferInfo vert_info;
    vert_info.buffer = asset_buffer.buffer;
    vert_info.offset = buffer_offsets[1];
    vert_info.range = buffer_sizes[2];

    desc_updates.push_back({});
    DescHelper::storage(desc_updates[2], asset_set_draw_, &vert_info, 0);

    VkDescriptorBufferInfo mat_info;
    mat_info.buffer = asset_buffer.buffer;
    mat_info.offset = buffer_offsets[3];
    mat_info.range = buffer_sizes[4];

    desc_updates.push_back({});
    DescHelper::storage(desc_updates[3], asset_set_draw_, &mat_info, 1);

    material_textures_ = loadTextures(dev, alloc, render_queue_, textures);

    DynArray<VkDescriptorImageInfo> tx_infos(material_textures_.size()+1);
    for (auto &tx : material_textures_) {
        tx_infos.push_back({
                VK_NULL_HANDLE,
                tx.view,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                });
    }

    if (material_textures_.size())
    {
        desc_updates.push_back({});
        DescHelper::textures(desc_updates[4], asset_set_mat_tex_, tx_infos.data(), tx_infos.size(), 0);
    }

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
//zl implement zoe lynch implement zoe lynch 5
float getTimeInSeconds() {
    static auto startTime = std::chrono::high_resolution_clock::now(); // Static variable to store start time

    auto currentTime = std::chrono::high_resolution_clock::now(); // Get the current time point

    // Calculate the duration since the start time in seconds
    std::chrono::duration<float> elapsed = currentTime - startTime;
    float seconds = elapsed.count();

    return seconds;
}


void Renderer::startFrame()
{
    // Extract the VkDevice from the render::vk::Device
    
    float currentTime = getTimeInSeconds();
    std::cout << "Current Time: " << currentTime << std::endl;
    
    // Pass the extracted VkDevice to updateUniformBuffer
    updateUniformBuffer(dev.hdl, timeBufferMemory_, currentTime);
    std::cout << "Exiting startFrame" << std::endl;
    
    std::cout << "Entering startFrame" << std::endl;
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    
    
}

void Renderer::updateUniformBuffer(VkDevice dev1, VkDeviceMemory timeBufferMemory, float currentTime)
{
    std::cout << "Entering updateUniformBuffer" << std::endl;
    void* data;
    VkResult result = dev.dt.mapMemory(dev1, timeBufferMemory, 0, sizeof(float), 0, &data);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to map memory: " << result << std::endl;
        return;
    }
    
    memcpy(data, &currentTime, sizeof(float));
    dev.dt.unmapMemory(dev1, timeBufferMemory);
    std::cout << "Exiting updateUniformBuffer" << std::endl;
}


static void packView(const Device &dev,
                     HostBuffer &view_staging_buffer,
                     const ViewerCam &cam,
                     uint32_t fb_width, uint32_t fb_height)
{
    PackedViewData *staging = (PackedViewData *)view_staging_buffer.ptr;

    math::Quat rotation =
        math::Quat::fromBasis(cam.right, cam.fwd, cam.up).inv();

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

// From GLM
math::Mat4x4 lookAt(
    math::Vector3 eye,
    math::Vector3 center,
    math::Vector3 up)
{
    math::Vector3 f = math::Vector3((center - eye).normalize());
    math::Vector3 s = math::Vector3((cross(f, up).normalize()));
    math::Vector3 u = math::Vector3(cross(s, f).normalize());

    math::Mat4x4 m = math::Mat4x4{};
    m.cols[0][0] = s.x;
    m.cols[1][0] = s.y;
    m.cols[2][0] = s.z;
    m.cols[0][1] = u.x;
    m.cols[1][1] = u.y;
    m.cols[2][1] = u.z;
    m.cols[0][2] =-f.x;
    m.cols[1][2] =-f.y;
    m.cols[2][2] =-f.z;
    m.cols[3][0] =-dot(s, eye);
    m.cols[3][1] =-dot(u, eye);
    m.cols[3][2] = dot(f, eye);
    m.cols[3][3] = 1.0f;
    return m;
}

static void packSky( const Device &dev,
                     HostBuffer &staging)
{
    SkyData *data = (SkyData *)staging.ptr;

    /* 
       Based on the values of Eric Bruneton's atmosphere model.
       We aren't going to calculate these manually come on (at least not yet)
    */

    /* 
       These are the irradiance values at the top of the Earth's atmosphere
       for the wavelengths 680nm (red), 550nm (green) and 440nm (blue) 
       respectively. These are in W/m2
       */
    data->solarIrradiance = math::Vector4{1.474f, 1.8504f, 1.91198f, 0.0f};
    // Angular radius of the Sun (radians)
    data->solarAngularRadius = 0.004675f;
    data->bottomRadius = 6360.0f / 2.0f;
    data->topRadius = 6420.0f / 2.0f;

    data->rayleighDensity.layers[0] =
        DensityLayer { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, {} };
    data->rayleighDensity.layers[1] =
        DensityLayer { 0.0f, 1.0f, -0.125f, 0.0f, 0.0f, {} };
    data->rayleighScatteringCoef =
        math::Vector4{0.005802f, 0.013558f, 0.033100f, 0.0f};

    data->mieDensity.layers[0] =
        DensityLayer { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, {} };
    data->mieDensity.layers[1] =
        DensityLayer { 0.0f, 1.0f, -0.833333f, 0.0f, 0.0f, {} };
    data->mieScatteringCoef = math::Vector4{0.003996f, 0.003996f, 0.003996f, 0.0f};
    data->mieExtinctionCoef = math::Vector4{0.004440f, 0.004440f, 0.004440f, 0.0f};

    data->miePhaseFunctionG = 0.8f;

    data->absorptionDensity.layers[0] =
        DensityLayer { 25.000000f, 0.000000f, 0.000000f, 0.066667f, -0.666667f, {} };
    data->absorptionDensity.layers[1] =
        DensityLayer { 0.000000f, 0.000000f, 0.000000f, -0.066667f, 2.666667f, {} };
    data->absorptionExtinctionCoef =
        math::Vector4{0.000650f, 0.001881f, 0.000085f, 0.0f};
    data->groundAlbedo = math::Vector4{0.050000f, 0.050000f, 0.050000f, 0.0f};
    data->muSunMin = -0.207912f;
    data->wPlanetCenter =
      math::Vector4{0.0f, 0.0f, -data->bottomRadius, 0.0f};
    data->sunSize = math::Vector4{
            0.0046750340586467079f, 0.99998907220740285f, 0.0f, 0.0f};

    staging.flush(dev);
}

static void packLighting(const Device &dev,
                         HostBuffer &light_staging_buffer,
                         const HeapArray<DirectionalLight> &lights)
{
    DirectionalLight *staging = (DirectionalLight *)light_staging_buffer.ptr;
    memcpy(staging, lights.data(), sizeof(DirectionalLight) * InternalConfig::maxLights);
    light_staging_buffer.flush(dev);
}

static void issueShadowGen(Device &dev,
                           Frame &frame,
                           Pipeline<1> &pipeline,
                           VkCommandBuffer draw_cmd,
                           uint32_t view_idx,
                           uint32_t max_views)
{
    {
        VkBufferMemoryBarrier compute_prepare = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            frame.renderInput.buffer,
            0, 
            (VkDeviceSize)frame.renderInputSize
        };

        dev.dt.cmdPipelineBarrier(draw_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                0, 0, nullptr, 1, &compute_prepare, 0, nullptr);
    }


    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.hdls[0]);

    shader::ShadowGenPushConst push_const = { view_idx };

    dev.dt.cmdPushConstants(draw_cmd,
                            pipeline.layout,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(shader::ShadowGenPushConst),
                            &push_const);

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.layout, 0, 1, &frame.shadowGenSet, 0, nullptr);

    uint32_t num_workgroups_x = utils::divideRoundUp(max_views, 32_u32);
    dev.dt.cmdDispatch(draw_cmd, num_workgroups_x, 1, 1);


    {
        VkBufferMemoryBarrier compute_prepare = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            frame.renderInput.buffer,
            0, 
            (VkDeviceSize)frame.renderInputSize
        };

        dev.dt.cmdPipelineBarrier(draw_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                0, 0, nullptr, 1, &compute_prepare, 0, nullptr);
    }
}

static void issueShadowBlurPass(vk::Device &dev, Frame &frame, Pipeline<1> &pipeline, VkCommandBuffer draw_cmd)
{
    { // Transition for compute
        array<VkImageMemoryBarrier, 2> compute_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.shadowFB.varianceAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_NONE,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.shadowFB.intermediate.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
        }};

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr, 0, nullptr,
                compute_prepare.size(), compute_prepare.data());
    }

    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.hdls[0]);

    shader::BlurPushConst push_const;
    push_const.isVertical = 0;

    dev.dt.cmdPushConstants(draw_cmd, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_const), &push_const);

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.layout, 0, 1, &frame.shadowBlurSet, 0, nullptr);

    uint32_t num_workgroups_x = utils::divideRoundUp(InternalConfig::shadowMapSize, 32_u32);
    uint32_t num_workgroups_y = utils::divideRoundUp(InternalConfig::shadowMapSize, 32_u32);

    dev.dt.cmdDispatch(draw_cmd, num_workgroups_x, num_workgroups_y, 1);

    { // Transition for compute
        array<VkImageMemoryBarrier, 2> compute_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.shadowFB.varianceAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.shadowFB.intermediate.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
        }};

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr, 0, nullptr,
                compute_prepare.size(), compute_prepare.data());
    }

    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.hdls[0]);

    push_const.isVertical = 1;

    dev.dt.cmdPushConstants(draw_cmd, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_const), &push_const);

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.layout, 0, 1, &frame.shadowBlurSet, 0, nullptr);

    dev.dt.cmdDispatch(draw_cmd, num_workgroups_x, num_workgroups_y, 1);

    { // Transition for compute
        array<VkImageMemoryBarrier, 2> compute_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.shadowFB.varianceAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.shadowFB.intermediate.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
        }};

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr, 0, nullptr,
                compute_prepare.size(), compute_prepare.data());
    }
}

static void issueLightingPass(vk::Device &dev, Frame &frame, Pipeline<1> &pipeline, VkCommandBuffer draw_cmd, const ViewerCam &cam, uint32_t view_idx)
{
    { // Transition for compute
        array<VkImageMemoryBarrier, 3> compute_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.colorAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.normalAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.positionAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
        }};

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr, 0, nullptr,
                compute_prepare.size(), compute_prepare.data());
    }

    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.hdls[0]);

    DeferredLightingPushConst push_const = {
        math::Vector4{ cam.fwd.x, cam.fwd.y, cam.fwd.z, 0.0f },
        math::Vector4{ cam.position.x, cam.position.y, cam.position.z, 0.0f },
        math::toRadians(cam.fov), 20.0f, 50.0f, view_idx
    };

    dev.dt.cmdPushConstants(draw_cmd, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DeferredLightingPushConst), &push_const);

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.layout, 0, 1, &frame.lightingSet, 0, nullptr);

    uint32_t num_workgroups_x = utils::divideRoundUp(frame.fb.colorAttachment.width, 32_u32);
    uint32_t num_workgroups_y = utils::divideRoundUp(frame.fb.colorAttachment.height, 32_u32);

    dev.dt.cmdDispatch(draw_cmd, num_workgroups_x, num_workgroups_y, 1);

    { // Transition for compute
        array<VkImageMemoryBarrier, 3> compute_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.colorAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.normalAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.positionAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
        }};

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0,
                0, nullptr, 0, nullptr,
                compute_prepare.size(), compute_prepare.data());
    }
}

void Renderer::render(const ViewerCam &cam,
                      const FrameConfig &cfg)
{
    static uint64_t global_frame_no = 0;

    Frame &frame = frames_[cur_frame_];
    uint32_t swapchain_idx = present_.acquireNext(dev, frame.swapchainReady);

    if (engine_interop_.renderInputCPU.has_value()) {
        // Need to flush engine input state before copy
        engine_interop_.renderInputCPU->flush(dev);
    }

    if (engine_interop_.voxelInputCPU.has_value()) {
        // Need to flush engine input state before copy
        engine_interop_.voxelInputCPU->flush(dev);
    }

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
    
    /*dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &timeBufferDescriptorSet, 0, nullptr);*/

    packLighting(dev, frame.lightStaging, lights_);
    VkBufferCopy light_copy {
        .srcOffset = 0,
        .dstOffset = frame.lightOffset,
        .size = sizeof(DirectionalLight) * InternalConfig::maxLights
    };

    dev.dt.cmdCopyBuffer(draw_cmd, frame.lightStaging.buffer,
                         frame.renderInput.buffer,
                         1, &light_copy);

    packSky(dev, frame.skyStaging);

    VkBufferCopy sky_copy {
        .srcOffset = 0,
        .dstOffset = frame.skyOffset,
        .size = sizeof(SkyData)
    };

    dev.dt.cmdCopyBuffer(draw_cmd, frame.skyStaging.buffer,
                         frame.renderInput.buffer,
                         1, &sky_copy);

    dev.dt.cmdFillBuffer(draw_cmd, frame.renderInput.buffer,
                         frame.drawCountOffset, sizeof(uint32_t), 0);

    VkDeviceSize world_view_byte_offset = engine_interop_.viewBaseOffset +
        cfg.worldIDX * engine_interop_.maxViewsPerWorld *
        sizeof(PackedViewData);

    VkBufferCopy view_data_copy {
        .srcOffset = world_view_byte_offset,
        .dstOffset = frame.simViewOffset,
        .size = sizeof(PackedViewData) * engine_interop_.maxViewsPerWorld,
    };

    dev.dt.cmdCopyBuffer(draw_cmd,
                         engine_interop_.renderInputHdl,
                         frame.renderInput.buffer,
                         1, &view_data_copy);
    

    issueShadowGen(dev, frame, shadow_gen_, draw_cmd,
                   cfg.viewIDX, engine_interop_.maxViewsPerWorld);

    const uint32_t num_voxels = this->voxel_config_.xLength * this->voxel_config_.yLength * this->voxel_config_.zLength;
    

    if (num_voxels > 0) {
        dev.dt.cmdFillBuffer(draw_cmd, frame.voxelVBO.buffer,
            0, sizeof(float) * num_voxels * 6 * 4 * 8, 0);

        VkBufferCopy voxel_copy = {
                .srcOffset = 0,
                .dstOffset = 0,
                .size = num_voxels * sizeof(int32_t),
        };

        dev.dt.cmdCopyBuffer(draw_cmd,
            engine_interop_.voxelHdl,
            frame.voxelData.buffer,
            1, &voxel_copy);

        issueVoxelGen(dev, frame, voxel_mesh_gen_, draw_cmd, cfg.viewIDX, engine_interop_.maxInstancesPerWorld, voxel_config_);
    }
    
    

    uint32_t num_instances =
        engine_interop_.bridge.numInstances[cfg.worldIDX];

    if (num_instances > 0) {
        VkDeviceSize world_instance_byte_offset = sizeof(PackedInstanceData) *
            cfg.worldIDX * engine_interop_.maxInstancesPerWorld;

        VkBufferCopy instance_copy = {
            .srcOffset = world_instance_byte_offset,
            .dstOffset = frame.instanceOffset,
            .size = sizeof(PackedInstanceData) * num_instances,
        };

        dev.dt.cmdCopyBuffer(draw_cmd,
                             engine_interop_.renderInputHdl,
                             frame.renderInput.buffer,
                             1, &instance_copy);

        dev.dt.cmdFillBuffer(draw_cmd,
            frame.renderInput.buffer,
            frame.drawCmdOffset,
            sizeof(VkDrawIndexedIndirectCommand) * num_instances * 10,
            0);

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
            .dstAccessMask = 
                VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                VK_ACCESS_SHADER_READ_BIT,
        };

        dev.dt.cmdPipelineBarrier(draw_cmd,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                                    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
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



    { // Shadow pass
        dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                object_shadow_draw_.hdls[0]);
        dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object_shadow_draw_.layout, 0, 1, &timeBufferDescriptorSet_, 0, nullptr);


        std::array draw_descriptors {
            frame.drawShaderSet,
            asset_set_draw_,
        };

        dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                object_shadow_draw_.layout, 0,
                draw_descriptors.size(),
                draw_descriptors.data(),
                0, nullptr);

        DrawPushConst draw_const {
            (uint32_t)cfg.viewIDX,
        };

        dev.dt.cmdPushConstants(draw_cmd, object_shadow_draw_.layout,
                VK_SHADER_STAGE_VERTEX_BIT |
                VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                sizeof(DrawPushConst), &draw_const);

        dev.dt.cmdBindIndexBuffer(draw_cmd, loaded_assets_[0].buf.buffer,
                loaded_assets_[0].idxBufferOffset,
                VK_INDEX_TYPE_UINT32);
        /*dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &timeBufferDescriptorSet, 0, nullptr);*/

        VkViewport viewport {
            0,
                0,
                (float)InternalConfig::shadowMapSize,
                (float)InternalConfig::shadowMapSize,
                0.f,
                1.f,
        };

        dev.dt.cmdSetViewport(draw_cmd, 0, 1, &viewport);

        VkRect2D scissor {
            { 0, 0 },
                { InternalConfig::shadowMapSize, InternalConfig::shadowMapSize },
        };

        dev.dt.cmdSetScissor(draw_cmd, 0, 1, &scissor);

        VkRenderPassBeginInfo render_pass_info;
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_info.pNext = nullptr;
        render_pass_info.renderPass = shadow_pass_;
        render_pass_info.framebuffer = frame.shadowFB.hdl;
        render_pass_info.clearValueCount = fb_shadow_clear_.size();
        render_pass_info.pClearValues = fb_shadow_clear_.data();
        render_pass_info.renderArea.offset = {
            0, 0,
        };
        render_pass_info.renderArea.extent = {
            InternalConfig::shadowMapSize, InternalConfig::shadowMapSize,
        };

        dev.dt.cmdBeginRenderPass(draw_cmd, &render_pass_info,
                VK_SUBPASS_CONTENTS_INLINE);

        dev.dt.cmdDrawIndexedIndirect(draw_cmd,
                frame.renderInput.buffer,
                frame.drawCmdOffset,
                num_instances * 10,
                sizeof(DrawCmd));

        dev.dt.cmdEndRenderPass(draw_cmd);

        issueShadowBlurPass(dev, frame, blur_, draw_cmd);

#if 1
        array<VkImageMemoryBarrier, 1> finish_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.shadowFB.depthAttachment.image,
                {
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    0, 1, 0, 1
                },
            }
        }};

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr, 0, nullptr,
                finish_prepare.size(), finish_prepare.data());
#endif
    }

    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           object_draw_.hdls[0]);
    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object_draw_.layout, 0, 1, &timeBufferDescriptorSet_, 0, nullptr);


    std::array draw_descriptors {
        frame.drawShaderSet,
        asset_set_draw_,
        asset_set_mat_tex_
    };

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 object_draw_.layout, 0,
                                 draw_descriptors.size(),
                                 draw_descriptors.data(),
                                 0, nullptr);
    
    /*dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     pipelineLayout, // Use the layout here
                                     0,
                                     1,
                                     &timeBufferDescriptorSet,
                                     0, nullptr);*/
    

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
                                  num_instances * 10,
                                  sizeof(DrawCmd));

    if (num_voxels > 0) {
        dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            voxel_draw_.hdls[0]);

        std::array voxel_draw_descriptors {
            frame.voxelDrawSet,
                asset_set_mat_tex_
        };

        dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            voxel_draw_.layout, 0,
            voxel_draw_descriptors.size(),
            voxel_draw_descriptors.data(),
            0, nullptr);

        DrawPushConst voxel_draw_const{
            (uint32_t)cfg.viewIDX,
        };

        dev.dt.cmdPushConstants(draw_cmd, voxel_draw_.layout,
            VK_SHADER_STAGE_VERTEX_BIT |
            VK_SHADER_STAGE_FRAGMENT_BIT, 0,
            sizeof(DrawPushConst), &voxel_draw_const);

        dev.dt.cmdBindIndexBuffer(draw_cmd, frame.voxelIndexBuffer.buffer,
            0,
            VK_INDEX_TYPE_UINT32);
        dev.dt.cmdDrawIndexed(draw_cmd, static_cast<uint32_t>(num_voxels * 6 * 6),
            1, 0, 0, 0);
    }

    dev.dt.cmdEndRenderPass(draw_cmd);

    { // Issue deferred lighting pass - separate function - this is becoming crazy
        issueLightingPass(dev, frame, deferred_lighting_, draw_cmd, cam, cfg.viewIDX);
    }

    bool prepare_screenshot = cfg.requestedScreenshot || (getenv("SCREENSHOT_PATH") && global_frame_no == 0);

    if (prepare_screenshot) {
        array<VkImageMemoryBarrier, 1> prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
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

        VkBufferMemoryBarrier buffer_prepare = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_NONE,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = screenshot_buffer_->buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE
        };

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr, 1, &buffer_prepare,
                prepare.size(), prepare.data());

        VkBufferImageCopy region = {
            .bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .imageOffset = {},
            .imageExtent = {frame.fb.colorAttachment.width, frame.fb.colorAttachment.height, 1}
        };

        dev.dt.cmdCopyImageToBuffer(draw_cmd, frame.fb.colorAttachment.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    screenshot_buffer_->buffer, 1, &region);

        prepare[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        prepare[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        prepare[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        prepare[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr, 0, nullptr,
                prepare.size(), prepare.data());
    }
    else {
        array<VkImageMemoryBarrier, 1> prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0,
                0, nullptr, 0, nullptr,
                prepare.size(), prepare.data());
    }

    render_pass_info.framebuffer = frame.imguiFBO.hdl;
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

    if (prepare_screenshot) {
        const char *ss_path = cfg.screenshotFilePath;

        if (!cfg.requestedScreenshot) {
            ss_path = getenv("SCREENSHOT_PATH");
        }

        dev.dt.deviceWaitIdle(dev.hdl);

        void *pixels = screenshot_buffer_->ptr;

        std::string dst_file = ss_path;
        int ret = stbi_write_bmp(dst_file.c_str(), frame.fb.colorAttachment.width, frame.fb.colorAttachment.height, 4,
            pixels);

        if (ret) {
            printf("Wrote %s\n", dst_file.c_str());
        }
    }

    global_frame_no++;
}

void Renderer::waitForIdle()
{
    dev.dt.deviceWaitIdle(dev.hdl);
}

const VizECSBridge * Renderer::getBridgePtr() const
{
    return engine_interop_.gpuBridge ?
        engine_interop_.gpuBridge : &engine_interop_.bridge;
}


}
