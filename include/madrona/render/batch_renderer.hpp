#pragma once

#include <memory>

#include <madrona/types.hpp>
#include <madrona/importer.hpp>
#include <madrona/render/batch_renderer_system.hpp>

namespace madrona {
namespace render {

struct BatchRendererECSBridge;

struct BatchRenderer {
    struct Impl;
    Impl *impl;

    struct Config {
        int gpuID;
        uint32_t renderWidth;
        uint32_t renderHeight;
        uint32_t numWorlds;
        uint32_t maxViewsPerWorld;
        uint32_t maxInstancesPerWorld;
        uint32_t maxObjects;
        BatchRendererECSBridge *bridge;
    };

    static BatchRendererECSBridge *makeECSBridge();
    static BatchRenderer make(const Config &cfg);

    ~BatchRenderer();

    CountT loadObjects(Span<const imp::SourceObject> objs);
    // RendererInterface getInterface() const;

    uint8_t *rgbPtr() const;
    float *depthPtr() const;

    void render();
};

}
}
