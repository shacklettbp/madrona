#pragma once

#include "vk/memory.hpp"
#include <memory>
#include <madrona/importer.hpp>
#include <madrona/render/vk/backend.hpp>
#include <madrona/render/vk/device.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include "vk/cuda_interop.hpp"
#endif

#include "ecs_interop.hpp"
#include "render_common.hpp"

#include <madrona/render/render_mgr.hpp>

namespace madrona::render {

struct RenderContext;

struct LayeredTarget {
    // Contains a uint for triangle ID and another for instance ID
    render::vk::LocalImage vizBuffer;
    VkImageView vizBufferView;

    // Depth
    render::vk::LocalImage depth;
    VkImageView depthView;

    uint32_t numViews;

    VkDescriptorSet lightingSet;

    uint32_t pixelWidth;
    uint32_t pixelHeight;

    uint32_t viewWidth;
    uint32_t viewHeight;
};

struct BatchRenderInfo {
    uint32_t numViews;
    uint32_t numInstances;
    uint32_t numWorlds;
    uint32_t numLights;
};

struct BatchImportedBuffers {
    render::vk::LocalBuffer views;
    render::vk::LocalBuffer viewOffsets;

    render::vk::LocalBuffer instances;
    render::vk::LocalBuffer instanceOffsets;

    render::vk::LocalBuffer lights;
    render::vk::LocalBuffer lightOffsets;
};

struct BatchRenderer {
    struct Impl;
    std::unique_ptr<Impl> impl;

    bool didRender;

    struct Config {
        bool enableBatchRenderer;

        RenderManager::Config::RenderMode renderMode;
        uint32_t renderWidth;
        uint32_t renderHeight;
        uint32_t numWorlds;
        uint32_t maxViewsPerWorld;
        uint32_t maxInstancesPerWorld;
        uint32_t numFrames;
    };

    BatchRenderer(const Config& cfg,
                  RenderContext &rctx);

    ~BatchRenderer();
    void importCudaData(VkCommandBuffer);

    void prepareForRendering(BatchRenderInfo info,
                             EngineInterop *interop);

    void renderViews(BatchRenderInfo info,
                     const DynArray<AssetData> &loaded_assets,
                     EngineInterop *interop,
                     RenderContext &rctx);

    BatchImportedBuffers &getImportedBuffers(uint32_t frame_id);

    const vk::LocalBuffer & getRGBBuffer() const;
    const vk::LocalBuffer & getDepthBuffer() const;

    // Get the semaphore that the viewer renderer has to wait on.
    // This is either going to be the semaphore from prepareForRendering,
    // or it's the one from renderViews.
    VkSemaphore getLatestWaitSemaphore();

    const uint8_t * getRGBCUDAPtr() const;
    const float * getDepthCUDAPtr() const;
};

}
