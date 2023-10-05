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

namespace madrona::viz {

struct BatchRenderInfo {
    uint32_t numViews;
    uint32_t numInstances;
};

struct BatchImportedBuffers {
    render::vk::LocalBuffer views;
    render::vk::LocalBuffer instances;
    render::vk::LocalBuffer instanceOffsets;
};

struct BatchRendererProto {
    struct Impl;
    std::unique_ptr<Impl> impl;

    struct Config {
        int gpuID;
        uint32_t renderWidth;
        uint32_t renderHeight;
        uint32_t numWorlds;
        uint32_t maxViewsPerWorld;
        uint32_t maxInstancesPerWorld;
    };

    BatchRendererProto(const Config &cfg,
                       render::vk::Device &dev,
                       render::vk::MemoryAllocator &mem,
                       VkPipelineCache pipeline_cache);

    ~BatchRendererProto();

    void importCudaData(VkCommandBuffer);
    void renderViews(VkCommandBuffer);

    BatchImportedBuffers &getImportedBuffers(uint32_t frame_id);
};

}
