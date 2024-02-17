#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>

namespace madrona::render {

// This will be attached to any entity that wants to be a viewer
struct RenderCamera {
    Entity cameraEntity;

    // 1.0 / tanf(fovy * 0.5)
    float fovScale;
    float zNear;

    math::Vector3 cameraOffset;
};

// This will be attached to any renderable entity
struct Renderable {
    Entity renderEntity;
};

// This will be attached to any renderable entity
using MortonCode = uint32_t;

struct RenderECSBridge;

namespace RenderingSystem {
    void registerTypes(ECSRegistry &registry,
                       const RenderECSBridge *bridge);

    TaskGraphNodeID setupTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);

    void init(Context &ctx,
              const RenderECSBridge *bridge);

    uint32_t * getVoxelPtr(Context &ctx);

    void makeEntityRenderable(Context &ctx,
                              Entity e);

    void attachEntityToView(Context &ctx,
                            Entity e,
                            float vfov_degrees,
                            float z_near,
                            const math::Vector3 &camera_offset);

    // Need to call these before destroying entities
    void cleanupViewingEntity(Context &ctx,
                              Entity e);
    void cleanupRenderableEntity(Context &ctx,
                                 Entity e);
};

}
