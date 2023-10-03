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

struct BatchRendererInterop {
    // For the batch renderer prototype
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

    CountT loadObjects(Span<const imp::SourceObject> objs);
    // RendererInterface getInterface() const;

    uint8_t *rgbPtr() const;
    float *depthPtr() const;

    void render();
};

}
