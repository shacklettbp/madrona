#pragma once

#include <memory>
#include <madrona/math.hpp>
#include <madrona/importer.hpp>
#include <madrona/exec_mode.hpp>
#include <madrona/viz/common.hpp>

namespace madrona::render {

// This encapsulates all rendering operations that the engine could require
struct RenderContext {
    struct Config {
        int gpuID; // Set to 0

        // Width and height of the batch renderer output
        uint32_t viewWidth;
        uint32_t viewHeight;

        uint32_t numWorlds;
        uint32_t maxViewsPerWorld;

        uint32_t maxInstancesPerWorld;
        uint32_t defaultSimTickRate;
        ExecMode execMode;

        bool renderViewer;
        uint32_t viewerWidth;
        uint32_t viewerHeight;

        VoxelConfig voxelCfg;
    };

    struct Impl;
    std::unique_ptr<Impl> impl;

    RenderContext(const Config &cfg);
    ~RenderContext();

    // Before anything rendering happens, objects must be loaded,
    // and lighting must be configured.
    CountT loadObjects(Span<const imp::SourceObject> objs,
                       Span<const imp::SourceMaterial> mats,
                       Span<const imp::SourceTexture> textures);

    void configureLighting(Span<const LightConfig> lights);



    // Processes the ECS's output in order to be ready for rendering.
    void prepareRender();

    // Draw the batched output for all worlds
    void batchedRender();

    // Returns output from the flycam rendering
    ViewerFrame *renderViewer(const ViewerInput &input);
};
    
}
