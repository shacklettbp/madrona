#pragma once

#include <madrona/render/vk/backend.hpp>
#include <madrona/render/vk/device.hpp>

#include <madrona/importer.hpp>

#include "vk/memory.hpp"
#include "vk/descriptors.hpp"
#include "vk/utils.hpp"

#include "../mw/render/interop.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace madrona::viz {

struct Swapchain {
    VkSwapchainKHR hdl;
    uint32_t width;
    uint32_t height;
};

struct Window {
    GLFWwindow *platformWindow;
    VkSurfaceKHR surface;

    uint32_t width;
    uint32_t height;
};

class PresentationState {
public:
    static PFN_vkGetInstanceProcAddr init();
    static std::vector<const char *> getInstanceExtensions();
    static VkSurfaceFormatKHR selectSwapchainFormat(const render::vk::Backend &backend,
                                                    VkPhysicalDevice phy,
                                                    VkSurfaceKHR surface);

    PresentationState(const render::vk::Backend &backend,
                      const render::vk::Device &dev,
                      const Window &window,
                      uint32_t num_frames_inflight,
                      bool need_immediate);

    void forceTransition(const render::vk::Device &dev,
                         const render::vk::QueueState &present_queue,
                         uint32_t qf_idx);

    uint32_t acquireNext(const render::vk::Device &dev,
                         VkSemaphore signal_sema);

    VkImage getImage(uint32_t idx) const;
    uint32_t numSwapchainImages() const;

    void present(const render::vk::Device &dev, uint32_t swapchain_idx,
                 const render::vk::QueueState &present_queue,
                 uint32_t num_wait_semas,
                 const VkSemaphore *wait_semas);

private:
    Swapchain swapchain_;
    DynArray<VkImage> swapchain_imgs_;
};

template <size_t N>
struct Pipeline {
    render::vk::PipelineShaders shader;
    VkPipelineLayout layout;
    std::array<VkPipeline, N> hdls;
    render::vk::FixedDescriptorPool descPool;
};

struct Framebuffer {
    render::vk::LocalImage colorAttachment;
    render::vk::LocalImage depthAttachment;
    VkImageView colorView;
    VkImageView depthView;
    VkFramebuffer hdl;
};

struct Frame {
    Framebuffer fb;
    VkCommandPool cmdPool;
    VkCommandBuffer drawCmd;
    VkFence cpuFinished;
    VkSemaphore renderFinished;
    VkSemaphore swapchainReady;

    render::vk::HostBuffer viewStaging;
    render::vk::LocalBuffer renderInput;
    uint32_t viewOffset;
    uint32_t drawCmdOffset;
    uint32_t drawCountOffset;
    uint32_t instanceOffset;
    uint32_t maxDraws;

    VkDescriptorSet cullShaderSet;
    VkDescriptorSet drawShaderSet;
};

struct ViewerCam {
    math::Vector3 position;
    math::Vector3 view;
    math::Vector3 up;
    math::Vector3 right;

    bool perspective = true;
    float fov = 60.f;
    float orthoHeight = 5.f;
    math::Vector2 mousePrev {0.f, 0.f};
};

struct AssetData {
    render::vk::LocalBuffer buf;
    uint32_t idxBufferOffset;
};

struct EngineInterop {
    render::vk::HostBuffer renderInputStaging;
    render::RendererBridge bridge;
    uint32_t maxViewsPerWorld;
    uint32_t maxInstancesPerWorld;
};

class Renderer {
public:
    struct FrameConfig {
        float lineWidth = 2.f;
        bool linesNoDepthTest = false;
    };

    Renderer(uint32_t gpu_id,
             uint32_t img_width,
             uint32_t img_height,
             uint32_t num_worlds,
             uint32_t max_views_per_world,
             uint32_t max_instances_per_world);
    Renderer(const Renderer &) = delete;

    CountT loadObjects(Span<const imp::SourceObject> objs);

    void waitUntilFrameReady();
    void startFrame();
    void render(const ViewerCam &cam,
                const FrameConfig &cfg,
                CountT world_idx);

    void waitForIdle();

    const render::RendererBridge & getBridgeRef() const;

    render::vk::Backend backend;
    Window window;
    render::vk::Device dev;
    render::vk::MemoryAllocator alloc;

private:
    VkQueue render_queue_;
    VkQueue transfer_queue_;
    VkQueue compute_queue_;
    VkQueue render_transfer_queue_;
    VkQueue compute_transfer_queue_;

    // Fixme remove
    render::vk::QueueState transfer_wrapper_;
    render::vk::QueueState render_transfer_wrapper_;
    render::vk::QueueState present_wrapper_;

    uint32_t fb_width_;
    uint32_t fb_height_;
    std::array<VkClearValue, 2> fb_clear_;
    PresentationState present_;
    VkPipelineCache pipeline_cache_;
    VkSampler repeat_sampler_;
    VkSampler clamp_sampler_;
    VkRenderPass render_pass_;
    VkRenderPass gui_render_pass_;
    Pipeline<1> instance_cull_;
    Pipeline<1> object_draw_;

    render::vk::FixedDescriptorPool asset_desc_pool_cull_;
    render::vk::FixedDescriptorPool asset_desc_pool_draw_;
    VkDescriptorSet asset_set_cull_;
    VkDescriptorSet asset_set_draw_;

    VkCommandPool load_cmd_pool_;
    VkCommandBuffer load_cmd_;
    VkFence load_fence_;

    EngineInterop engine_interop_;

    uint32_t cur_frame_;
    DynArray<Frame> frames_;
    DynArray<AssetData> loaded_assets_;
};

}
