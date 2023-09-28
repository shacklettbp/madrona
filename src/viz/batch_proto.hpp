#pragma once

#include "vk/memory.hpp"
#include <memory>
#include <madrona/importer.hpp>
#include <madrona/viz/interop.hpp>
#include <madrona/render/vk/backend.hpp>
#include <madrona/render/vk/device.hpp>

namespace madrona::viz {

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
        BatchRendererECSBridge *bridge;
    };

    BatchRendererProto(const Config &cfg,
                       render::vk::Device &dev,
                       render::vk::MemoryAllocator &mem,
                       VkPipelineCache pipeline_cache);

    static BatchRendererECSBridge *makeECSBridge();

    ~BatchRendererProto();

    CountT loadObjects(Span<const imp::SourceObject> objs);
    // RendererInterface getInterface() const;

    uint8_t *rgbPtr() const;
    float *depthPtr() const;

    void render();
};

}
