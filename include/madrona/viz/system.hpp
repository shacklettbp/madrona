#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>

namespace madrona::viz {

// This will be attached to any entity that wants to be a viewer
struct VizCamera {
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

struct VizECSBridge;

struct VizRenderingSystem {
    static void registerTypes(ECSRegistry &registry,
                              const VizECSBridge *bridge);

    static TaskGraphNodeID setupTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);

    static void reset(Context &ctx);

    static void init(Context &ctx,
                     const VizECSBridge *bridge);

    static uint32_t * getVoxelPtr(Context &ctx);

    static void markEpisode(Context &ctx);

    static void makeEntityRenderable(Context &ctx,
                                     Entity e);

    static void attachEntityToView(Context &ctx,
                                   Entity e,
                                   float vfov_degrees,
                                   float z_near,
                                   const math::Vector3 &camera_offset);

    // Need to call these before destroying entities
    static void cleanupViewingEntity(Context &ctx,
                                     Entity e);
    static void cleanupRenderableEntity(Context &ctx,
                                        Entity e);
};

}
