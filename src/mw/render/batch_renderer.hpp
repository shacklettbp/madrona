#pragma once

#include "interop.hpp"

#include <madrona/types.hpp>
#include <madrona/render/mw.hpp>
#include <madrona/importer.hpp>
#include <memory>

namespace madrona {
namespace render {

class BatchRenderer {
public:
    struct Config {
        int gpuID;
        uint32_t renderWidth;
        uint32_t renderHeight;
        uint32_t numWorlds;
        uint32_t maxViewsPerWorld;
        uint32_t maxInstancesPerWorld;
        uint32_t maxObjects;
        CameraMode cameraMode;
        ExecMode execMode;
    };

    BatchRenderer(const Config &cfg);
    BatchRenderer(BatchRenderer &&o);

    ~BatchRenderer();

    CountT loadObjects(Span<const imp::SourceObject> objs);

    RendererInterface getInterface() const;

    uint8_t * rgbPtr() const;
    float * depthPtr() const;

    void render();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}
