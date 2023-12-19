#pragma once

#include <madrona/render/vk/backend.hpp>
#include <madrona/render/vk/device.hpp>

#include <madrona/importer.hpp>

#include "vk/memory.hpp"
#include "vk/descriptors.hpp"
#include "vk/utils.hpp"

#include "interop.hpp"

#include <madrona/viz/viewer.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#ifdef MADRONA_CUDA_SUPPORT
#include "vk/cuda_interop.hpp"
#endif

#include "shader.hpp"

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
    static render::vk::Backend::LoaderLib init();
    static std::vector<const char *> getInstanceExtensions();
    static VkSurfaceFormatKHR selectSwapchainFormat(const render::vk::Backend &backend,
                                                    VkPhysicalDevice phy,
                                                    VkSurfaceKHR surface);

    PresentationState(const render::vk::Backend &backend,
                      const render::vk::Device &dev,
                      const Window &window,
                      uint32_t num_frames_inflight,
                      bool need_immediate);

    void destroy(const render::vk::Device &dev);

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
    HeapArray<VkImage> swapchain_imgs_;
};

template <size_t N>
struct Pipeline {
    render::vk::PipelineShaders shaders;
    VkPipelineLayout layout;
    std::array<VkPipeline, N> hdls;
    render::vk::FixedDescriptorPool descPool;
};

struct Framebuffer {
    render::vk::LocalImage colorAttachment;
    render::vk::LocalImage normalAttachment;
    render::vk::LocalImage positionAttachment;
    render::vk::LocalImage depthAttachment;
    VkImageView colorView;
    VkImageView normalView;
    VkImageView positionView;
    VkImageView depthView;
    VkFramebuffer hdl;
};

struct ShadowFramebuffer {
    render::vk::LocalImage varianceAttachment;
    render::vk::LocalImage intermediate;
    render::vk::LocalImage depthAttachment;

    VkImageView varianceView;
    VkImageView intermediateView;
    VkImageView depthView;
    VkFramebuffer hdl;
};

struct MaterialTexture {
    MaterialTexture(render::vk::LocalTexture &&src_image,
                    VkImageView src_view,
                    VkDeviceMemory src_backing)
        : image(std::move(src_image)), view(src_view), backing(src_backing)
    {
    }

    render::vk::LocalTexture image;
    VkImageView view;
    VkDeviceMemory backing;
};

struct Frame {
    Framebuffer fb;
    Framebuffer imguiFBO;

    ShadowFramebuffer shadowFB;

    VkCommandPool cmdPool;
    VkCommandBuffer drawCmd;
    VkFence cpuFinished;
    VkSemaphore renderFinished;
    VkSemaphore swapchainReady;

    render::vk::HostBuffer viewStaging;
    render::vk::HostBuffer lightStaging;
    // Don't need a shadow view staging because that will be done on the GPU.
    render::vk::HostBuffer skyStaging;
    render::vk::LocalBuffer renderInput;

    render::vk::LocalBuffer voxelVBO;
    render::vk::LocalBuffer voxelIndexBuffer;
    render::vk::LocalBuffer voxelData;

    int64_t renderInputSize;

    uint32_t cameraViewOffset;
    uint32_t simViewOffset;
    uint32_t drawCmdOffset;
    uint32_t drawCountOffset;
    uint32_t instanceOffset;
    uint32_t lightOffset;
    uint32_t shadowOffset;
    uint32_t skyOffset;
    uint32_t maxDraws;

    VkDescriptorSet cullShaderSet;
    VkDescriptorSet drawShaderSet;
    VkDescriptorSet lightingSet;
    VkDescriptorSet shadowGenSet;
    VkDescriptorSet shadowBlurSet;

    VkDescriptorSet voxelGenSet;
    VkDescriptorSet voxelDrawSet;
};

struct ViewerCam {
    math::Vector3 position;
    math::Vector3 fwd;
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
    Optional<render::vk::HostBuffer> renderInputCPU;
#ifdef MADRONA_CUDA_SUPPORT
    Optional<render::vk::DedicatedBuffer> renderInputGPU;
    Optional<render::vk::CudaImportedBuffer> renderInputCUDA;
#endif
    VkBuffer renderInputHdl;
    viz::VizECSBridge bridge;
    const viz::VizECSBridge *gpuBridge;
    uint32_t viewBaseOffset;
    uint32_t maxViewsPerWorld;
    uint32_t maxInstancesPerWorld;

    Optional<render::vk::HostBuffer> voxelInputCPU;
#ifdef MADRONA_CUDA_SUPPORT
    Optional<render::vk::DedicatedBuffer> voxelInputGPU;
    Optional<render::vk::CudaImportedBuffer> voxelInputCUDA;
#endif
    VkBuffer voxelHdl;
};

struct ImGuiRenderState {
    VkDescriptorPool descPool;
    VkRenderPass renderPass;
};

struct ShadowOffsets {
    render::vk::LocalTexture offsets;
    VkImageView view;
    VkDeviceMemory backing;

    uint32_t outerDimension;
    uint32_t filterSize;
};

struct Sky {
    render::vk::LocalTexture transmittance;
    render::vk::LocalTexture scattering;
    render::vk::LocalTexture singleMieScattering;
    render::vk::LocalTexture irradiance;

    VkImageView transmittanceView;
    VkImageView scatteringView;
    VkImageView mieView;
    VkImageView irradianceView;

    VkDeviceMemory transmittanceBacking;
    VkDeviceMemory scatteringBacking;
    VkDeviceMemory mieBacking;
    VkDeviceMemory irradianceBacking;

    math::Vector3 sunDirection;
    math::Vector3 white;
    math::Vector3 sunSize;
    float exposure;
};

class Renderer {
public:
    struct FrameConfig {
        uint32_t worldIDX;
        uint32_t viewIDX;
        bool requestedScreenshot;
        const char *screenshotFilePath;
    };

    Renderer(uint32_t gpu_id,
             uint32_t img_width,
             uint32_t img_height,
             uint32_t num_worlds,
             uint32_t max_views_per_world,
             uint32_t max_instances_per_world,
             bool gpu_input,
             VoxelConfig voxel_config);
    Renderer(const Renderer &) = delete;
    ~Renderer();

    CountT loadObjects(Span<const imp::SourceObject> objs,
                       Span<const imp::SourceMaterial> mats,
                       Span<const imp::SourceTexture> textures);

    void configureLighting(Span<const LightConfig> lights);

    void waitUntilFrameReady();
    void startFrame();
    void render(const ViewerCam &cam,
                const FrameConfig &cfg);

    void waitForIdle();

    const VizECSBridge * getBridgePtr() const;
   /* void startFrame() {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        float currentTime = getTimeInSeconds();
        updateUniformBuffer(currentTime); // Call the member function
    }*/
    void updateUniformBuffer(VkDevice dev, VkDeviceMemory timeBufferMemory, float currentTime);// Unmap the memory after writing
    



private:
    render::vk::Backend::LoaderLib loader_lib_;

public:
    render::vk::Backend backend;
    Window window;
    render::vk::Device dev;
    render::vk::MemoryAllocator alloc;

private:
    VkBuffer timeBuffer_;
    VkDeviceMemory timeBufferMemory_;
    VkDescriptorSetLayout timeBufferLayout_;
    VkDescriptorSet timeBufferDescriptorSet_;
    madrona::render::vk::InstanceDispatch instanceDispatch_;
    
    
    VkQueue render_queue_;

    // Fixme remove
    render::vk::QueueState present_wrapper_;

    uint32_t fb_width_;
    uint32_t fb_height_;
    std::array<VkClearValue, 4> fb_clear_;
    std::array<VkClearValue, 2> fb_shadow_clear_;
    std::array<VkClearValue, 2> fb_imgui_clear_;
    PresentationState present_;
    VkPipelineCache pipeline_cache_;
    VkSampler repeat_sampler_;
    VkSampler clamp_sampler_;
    VkRenderPass render_pass_;
    VkRenderPass shadow_pass_;
    ImGuiRenderState imgui_render_state_;

    Pipeline<1> instance_cull_;
    Pipeline<1> object_draw_;
    Pipeline<1> object_shadow_draw_;
    Pipeline<1> deferred_lighting_;
    Pipeline<1> shadow_gen_;
    Pipeline<1> blur_;
    Pipeline<1> voxel_mesh_gen_;
    Pipeline<1> voxel_draw_;

    render::vk::FixedDescriptorPool asset_desc_pool_cull_;
    render::vk::FixedDescriptorPool asset_desc_pool_draw_;
    render::vk::FixedDescriptorPool asset_desc_pool_mat_tx_;

    VkDescriptorSet asset_set_cull_;
    VkDescriptorSet asset_set_draw_;
    VkDescriptorSet asset_set_mat_tex_;

    VkCommandPool load_cmd_pool_;
    VkCommandBuffer load_cmd_;
    VkFence load_fence_;

    EngineInterop engine_interop_;

    HeapArray<shader::DirectionalLight> lights_;

    uint32_t cur_frame_;
    HeapArray<Frame> frames_;
    DynArray<AssetData> loaded_assets_;

    Sky sky_;

    DynArray<MaterialTexture> material_textures_;
    VoxelConfig voxel_config_;
    
    void createTimeBuffer();
    void createTimeBufferDescriptorSet();

    std::unique_ptr<render::vk::HostBuffer> screenshot_buffer_;
    
};

}
