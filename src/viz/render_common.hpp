#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <madrona/render/vk/backend.hpp>
#include <madrona/render/vk/device.hpp>

#include "vk/memory.hpp"
#include "vk/descriptors.hpp"
#include "vk/utils.hpp"

#include <madrona/viz/interop.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#ifdef MADRONA_CUDA_SUPPORT
#include "vk/cuda_interop.hpp"
#endif

#include "shader.hpp"

namespace madrona::render {

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
    static render::vk::Backend::LoaderLib init(bool make_window);
    static std::vector<const char *> getInstanceExtensions(bool make_window);
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

struct ViewerAppCfg {

};

struct ViewerFrame {

};
    
}
