#pragma once

#include <madrona/math.hpp>
#include <madrona/importer.hpp>
#include <madrona/exec_mode.hpp>

#include <madrona/render/common.hpp>
#include <madrona/render/ecs.hpp>

#include <memory>

namespace madrona::render {

struct RenderContext;

// This encapsulates all rendering operations that the engine could require
class RenderManager {
public:
    struct Config {
        enum RenderMode {
            Color,
            Depth,
        };

        bool enableBatchRenderer;

        RenderMode renderMode;

        // Width and height of the batch renderer output
        uint32_t agentViewWidth;
        uint32_t agentViewHeight;

        uint32_t numWorlds;
        uint32_t maxViewsPerWorld;

        uint32_t maxInstancesPerWorld;
        ExecMode execMode;

        VoxelConfig voxelCfg;
    };

    RenderManager(APIBackend *render_backend,
                  GPUDevice *dev,
                  const Config &cfg);
    RenderManager(RenderManager &&);
    ~RenderManager();

    // Before anything rendering happens, objects must be loaded,
    // and lighting must be configured.
    CountT loadObjects(Span<const imp::SourceObject> objs,
                       Span<const imp::SourceMaterial> mats,
                       Span<const imp::SourceTexture> textures,
                       bool override_materials = false);

    void configureLighting(Span<const LightConfig> lights);

    const RenderECSBridge * bridge() const;
    inline RenderContext & renderContext() const;

    // Processes the ECS's output in order to be ready for rendering.
    void readECS();

    // Draw the batched output for all worlds
    void batchRender();

    const uint8_t * batchRendererRGBOut() const;
    const float * batchRendererDepthOut() const;

private:
    std::unique_ptr<RenderContext> rctx_;
};

}

#include "render_mgr.inl"
