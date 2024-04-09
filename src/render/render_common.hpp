#pragma once

#include <madrona/render/vk/backend.hpp>
#include <madrona/render/vk/device.hpp>
#include <madrona/dyn_array.hpp>

#include <cstdint>

#include "vk/memory.hpp"
#include "vk/descriptors.hpp"
#include "vk/utils.hpp"

#ifdef MADRONA_VK_CUDA_SUPPORT
#include "vk/cuda_interop.hpp"
#endif

#include "shader.hpp"
#include "ecs_interop.hpp"

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

struct AssetData {
    render::vk::LocalBuffer buf;
    uint32_t idxBufferOffset;

    // This is a descriptor set which just contains the buffer
    VkDescriptorSet indexBufferSet;
};

struct EngineInterop {
    Optional<render::vk::HostBuffer> viewsCPU;
    Optional<render::vk::HostBuffer> viewOffsetsCPU;

    Optional<render::vk::HostBuffer> instancesCPU;
    Optional<render::vk::HostBuffer> instanceOffsetsCPU;

    Optional<render::vk::HostBuffer> aabbCPU;

#ifdef MADRONA_VK_CUDA_SUPPORT
    Optional<render::vk::DedicatedBuffer> viewsGPU;
    Optional<render::vk::DedicatedBuffer> viewOffsetsGPU;

    Optional<render::vk::DedicatedBuffer> instancesGPU;
    Optional<render::vk::DedicatedBuffer> instanceOffsetsGPU;

    Optional<render::vk::CudaImportedBuffer> viewsCUDA;
    Optional<render::vk::CudaImportedBuffer> viewOffsetsCUDA;

    Optional<render::vk::CudaImportedBuffer> instancesCUDA;
    Optional<render::vk::CudaImportedBuffer> instanceOffsetsCUDA;

    Optional<render::vk::DedicatedBuffer> aabbGPU;
    Optional<render::vk::CudaImportedBuffer> aabbCUDA;
#endif

    VkBuffer viewsHdl;
    VkBuffer viewOffsetsHdl;

    VkBuffer instancesHdl;
    VkBuffer instanceOffsetsHdl;

    VkBuffer aabbHdl;

    RenderECSBridge bridge;
    const RenderECSBridge *gpuBridge;

    uint32_t maxViewsPerWorld;
    uint32_t maxInstancesPerWorld;

    Optional<render::vk::HostBuffer> voxelInputCPU;
#ifdef MADRONA_VK_CUDA_SUPPORT
    Optional<render::vk::DedicatedBuffer> voxelInputGPU;
    Optional<render::vk::CudaImportedBuffer> voxelInputCUDA;
#endif

    VkBuffer voxelHdl;

    uint32_t *iotaArrayInstancesCPU;
    uint32_t *iotaArrayViewsCPU;

    // We need the sorted instance world IDs in order to compute the instance offsets
    uint64_t *sortedInstanceWorldIDs;
    uint64_t *sortedViewWorldIDs;
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

void initCommonDrawPipelineInfo(
    VkPipelineVertexInputStateCreateInfo &vert_info,
    VkPipelineInputAssemblyStateCreateInfo &input_assembly_info,
    VkPipelineViewportStateCreateInfo &viewport_info,
    VkPipelineMultisampleStateCreateInfo &multisample_info,
    VkPipelineRasterizationStateCreateInfo &raster_info);
    
}
