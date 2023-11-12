#pragma once

#include <madrona/render/vk/backend.hpp>
#include <madrona/render/vk/device.hpp>

#include <madrona/importer.hpp>

#include "vk/memory.hpp"
#include "vk/descriptors.hpp"
#include "vk/utils.hpp"

#include <madrona/viz/interop.hpp>

#include <madrona/viz/viewer.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#ifdef MADRONA_CUDA_SUPPORT
#include "vk/cuda_interop.hpp"
#endif

#include "shader.hpp"

namespace madrona::render {

struct BatchRenderer;
    
}

namespace madrona::viz {

struct BatchRenderer;

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

// If we want to be able to get multiple pools
template <size_t N>
struct PipelineMP {
    render::vk::PipelineShaders shaders;
    VkPipelineLayout layout;
    std::array<VkPipeline, N> hdls;
    DynArray<render::vk::FixedDescriptorPool> descPools;
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

    // Contains everything
    render::vk::LocalBuffer renderInput;

    render::vk::LocalBuffer voxelVBO;
    render::vk::LocalBuffer voxelIndexBuffer;
    render::vk::LocalBuffer voxelData;

    int64_t renderInputSize;

    uint32_t cameraViewOffset;
    // We now store this in a separate buffer
    // uint32_t simViewOffset;
    uint32_t drawCmdOffset;
    uint32_t drawCountOffset;
    // We now store this in a separate buffer
    // uint32_t instanceOffset;
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

    // Contains a descriptor set for the sampler state and the final rendered output
    VkDescriptorSet batchOutputQuadSet;
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

    // This is a descriptor set which just contains the buffer
    VkDescriptorSet indexBufferSet;
};

struct EngineInterop {
    Optional<render::vk::HostBuffer> viewsCPU;
    Optional<render::vk::HostBuffer> instancesCPU;
    Optional<render::vk::HostBuffer> instanceOffsetsCPU;

#ifdef MADRONA_CUDA_SUPPORT
    Optional<render::vk::DedicatedBuffer> viewsGPU;
    Optional<render::vk::DedicatedBuffer> instancesGPU;
    Optional<render::vk::DedicatedBuffer> instanceOffsetsGPU;

    Optional<render::vk::CudaImportedBuffer> viewsCUDA;
    Optional<render::vk::CudaImportedBuffer> instancesCUDA;
    Optional<render::vk::CudaImportedBuffer> instanceOffsetsCUDA;
#endif

    VkBuffer viewsHdl;
    VkBuffer instancesHdl;
    VkBuffer instanceOffsetsHdl;

    viz::VizECSBridge bridge;
    const viz::VizECSBridge *gpuBridge;

    uint32_t maxViewsPerWorld;
    uint32_t maxInstancesPerWorld;

    Optional<render::vk::HostBuffer> voxelInputCPU;
#ifdef MADRONA_CUDA_SUPPORT
    Optional<render::vk::DedicatedBuffer> voxelInputGPU;
    Optional<render::vk::CudaImportedBuffer> voxelInputCUDA;
#endif

    VkBuffer voxelHdl;

    uint32_t *iotaArrayInstancesCPU;
    uint32_t *iotaArrayViewsCPU;

    // We need the sorted instance world IDs in order to compute the instance offsets
    uint64_t *sortedInstanceWorldIDs;
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
        uint32_t batchViewIDX;
        uint32_t overrideLightDir;
        math::Vector3 newLightDir;
        bool requestedScreenshot;
        const char *screenshotFilePath;
    };

    Renderer(uint32_t gpu_id,
             uint32_t img_width,
             uint32_t img_height,
             uint32_t batch_width,
             uint32_t batch_height,
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

    void renderViews(const FrameConfig &cfg, bool just_do_transition = false);

    void waitForIdle();

    const VizECSBridge * getBridgePtr() const;

private:
    render::vk::Backend::LoaderLib loader_lib_;

public:
    render::vk::Backend backend;
    Window window;
    render::vk::Device dev;
    render::vk::MemoryAllocator alloc;

private:
    VkQueue render_queue_;

    // Fixme remove
    render::vk::QueueState present_wrapper_;

    uint32_t fb_width_;
    uint32_t fb_height_;

    uint32_t br_width_;
    uint32_t br_height_;

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
    Pipeline<1> quad_draw_;

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

    HeapArray<render::shader::DirectionalLight> lights_;

    uint32_t cur_frame_;
    HeapArray<Frame> frames_;
    DynArray<AssetData> loaded_assets_;

    Sky sky_;

    DynArray<MaterialTexture> material_textures_;
    VoxelConfig voxel_config_;

    // This is just a prototype
    int gpu_id_;
    uint32_t num_worlds_;
    std::unique_ptr<BatchRenderer> br_proto_;

    VkDescriptorSetLayout asset_layout_;
    VkDescriptorSetLayout asset_tex_layout_;

    // This contains the vertex data buffer and the mesh data buffer
    VkDescriptorSetLayout asset_batch_lighting_layout_;

    VkDescriptorPool asset_pool_;
    VkDescriptorSet asset_set_tex_compute_;
    VkDescriptorSet asset_batch_lighting_set_;

    // This descriptor set contains information about the sky
    VkDescriptorSetLayout sky_data_layout_;
    VkDescriptorSet sky_data_set_;

    bool gpu_input_;

    // This is only used if we are on the CPU backend
    uint32_t *iota_array_;
    std::unique_ptr<render::vk::HostBuffer> screenshot_buffer_;
};

}
