#pragma once

#include <memory>
#include <madrona/math.hpp>
#include <madrona/importer.hpp>
#include <madrona/exec_mode.hpp>

namespace madrona::render {

// Required for rendering the viewer image
struct ViewerInput {
    // Which world to render in the flycam
    uint32_t worldIdx;

    // Camera world position/direction configuration
    math::Vector3 position;
    math::Vector3 forward;
    math::Vector3 up;
    math::Vector3 right;

    // Camera configuration
    bool usePerspective = true;
    float fov = 60.0f;
    float orthoHeight = 0.5f;
    math::Vector2 mousePrev = {0.0f, 0.0f};
};

// Passed out after the flycam image has been rendered
struct ViewerOutput;

// Configures an individual light source
struct LightConfig {
    bool isDirectional;

    // Used for direction or position depending on value of isDirectional
    math::Vector3 dir;
    math::Vector3 color;
};

// If voxel generation is to happen
struct VoxelConfig {
    uint32_t xLength;
    uint32_t yLength;
    uint32_t zLength;
};

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

    // Before anything rendering happens, objects must be loaded,
    // and lighting must be configured.
    CountT loadObjects(Span<const imp::SourceObject> objs,
                       Span<const imp::SourceMaterial> mats,
                       Span<const imp::SourceTexture> textures);

    void configureLighting(Span<const LightConfig> lights);



    // Draw the batched output for all worlds
    void batchedRender();

    // Returns output from the flycam rendering
    ViewerOutput *renderViewer(const ViewerInput &input);
};
    
}
