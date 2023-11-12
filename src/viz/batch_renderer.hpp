#pragma once

#include "vk/memory.hpp"
#include <memory>
#include <madrona/importer.hpp>
#include <madrona/viz/interop.hpp>
#include <madrona/render/vk/backend.hpp>
#include <madrona/render/vk/device.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include "vk/cuda_interop.hpp"
#endif

// #include "render_common.hpp"
#include "viewer_renderer.hpp"

namespace madrona::viz {

struct LayeredTarget {
    // Contains a uint for triangle ID and another for instance ID
    render::vk::LocalImage vizBuffer;
    VkImageView vizBufferView;

    // Depth
    render::vk::LocalImage depth;
    VkImageView depthView;

    render::vk::LocalImage output;
    VkImageView outputView;

    uint32_t layerCount;

    VkDescriptorSet lightingSet;
};

// A texture containing the view we want to visualize in the viewer
struct DisplayTexture {
    render::vk::LocalTexture tex;
    VkDeviceMemory mem;
    VkImageView view;
};

struct BatchRenderInfo {
    uint32_t numViews;
    uint32_t numInstances;
    uint32_t numWorlds;
};

struct BatchImportedBuffers {
    render::vk::LocalBuffer views;
    render::vk::LocalBuffer instances;
    render::vk::LocalBuffer instanceOffsets;
};

struct BatchRenderer {
    struct Impl;
    std::unique_ptr<Impl> impl;

    struct Config {
        int gpuID;
        uint32_t renderWidth;
        uint32_t renderHeight;
        uint32_t numWorlds;
        uint32_t maxViewsPerWorld;
        uint32_t maxInstancesPerWorld;
        uint32_t numFrames;
    };

    BatchRenderer(const Config& cfg,
        render::vk::Device& dev,
        render::vk::MemoryAllocator& mem,
        VkPipelineCache pipeline_cache,
        VkDescriptorSet asset_set_compute,
        VkDescriptorSet asset_set_draw,
        VkDescriptorSet asset_set_texture_mat,
        VkDescriptorSet asset_set_lighting,
        VkSampler repeat_sampler,
        VkQueue render_queue);

    ~BatchRenderer();
    void importCudaData(VkCommandBuffer);

    void renderViews(BatchRenderInfo info,
                     const DynArray<AssetData> &loaded_assets,
                     uint32_t view_idx,
                     EngineInterop *interop);

    void transitionOutputLayout();

    BatchImportedBuffers &getImportedBuffers(uint32_t frame_id);
    DisplayTexture &getDisplayTexture(uint32_t frame_id);
    LayeredTarget &getLayeredTarget(uint32_t frame_id);
    VkDescriptorSet getPBRSet(uint32_t frame_id);

    // Get the semaphore that the viewer renderer has to wait on
    VkSemaphore getLatestWaitSemaphore();
};

}
