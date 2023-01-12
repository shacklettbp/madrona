#pragma once

#include <madrona/types.hpp>
#include <madrona/render.hpp>
#include <madrona/importer.hpp>
#include <memory>

namespace madrona {
namespace render {

class BatchRenderer {
public:
    enum class CameraMode : uint32_t {
        Perspective,
        Lidar,
    };

    struct Config {
        int gpuID;
        uint32_t renderWidth;
        uint32_t renderHeight;
        uint32_t numWorlds;
        uint32_t maxViewsPerWorld;
        uint32_t maxInstancesPerWorld;
        uint32_t maxObjects;
        CameraMode cameraMode;
    };

    BatchRenderer(const Config &cfg);
    BatchRenderer(BatchRenderer &&o);

    ~BatchRenderer();

    CountT loadObjects(Span<const imp::SourceObject> objs);

    AccelStructInstance ** tlasInstancePtrs() const;
    uint64_t * objectsBLASPtr() const;
    void *viewDataPtr() const;

    uint8_t * rgbPtr() const;
    float * depthPtr() const;

    void render(const uint32_t *num_instances);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}
