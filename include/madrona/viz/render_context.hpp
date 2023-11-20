#pragma once

#include <memory>
#include <madrona/math.hpp>
#include <madrona/importer.hpp>
#include <madrona/exec_mode.hpp>
#include <madrona/viz/common.hpp>
#include <madrona/viz/interop.hpp>
#include <madrona/viz/viewer_controller.hpp>

namespace madrona::render {

// This encapsulates all rendering operations that the engine could require
struct RenderContext {
    struct Config {
        int gpuID; // Set to 0

        // Width and height of the batch renderer output
        bool enableBatchRenderer;
        uint32_t viewWidth;
        uint32_t viewHeight;

        uint32_t numWorlds;
        uint32_t maxViewsPerWorld;

        uint32_t maxInstancesPerWorld;
        uint32_t defaultSimTickRate;
        ExecMode execMode;

        bool enableViewer;
        uint32_t viewerWidth;
        uint32_t viewerHeight;

        VoxelConfig voxelCfg;
    };

    struct Impl;
    std::unique_ptr<Impl> impl;

    RenderContext(const Config &cfg);
    RenderContext(RenderContext &&);
    ~RenderContext();

    // Before anything rendering happens, objects must be loaded,
    // and lighting must be configured.
    CountT loadObjects(Span<const imp::SourceObject> objs,
                       Span<const imp::SourceMaterial> mats,
                       Span<const imp::SourceTexture> textures);

    void configureLighting(Span<const LightConfig> lights);

    const viz::VizECSBridge * getBridge();



    // Create the viewer controller
    viz::ViewerController makeViewerController(float camera_move_speed,
                                               math::Vector3 camera_pos,
                                               math::Quat camera_rot);



    // Processes the ECS's output in order to be ready for rendering.
    void prepareRender();

    // Draw the batched output for all worlds
    void batchedRender();

    // Returns output from the flycam rendering
    viz::ViewerFrame * renderViewer(const viz::ViewerInput &input);
};
    
}
