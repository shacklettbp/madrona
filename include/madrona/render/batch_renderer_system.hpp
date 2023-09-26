#pragma once

#include <madrona/math.hpp>
#include <madrona/taskgraph_builder.hpp>

namespace madrona::render {

// Make sure that this becomes packed in the future.
struct alignas(16) PerspectiveCameraData {
    math::Vector4 position;
    math::Quat rotation;
    float xScale;
    float yScale;
    float zNear;
    int32_t viewIDX;
    int32_t worldIDX;
};

// Instance data that is needed to render an object
struct alignas(16) InstanceData {
    math::Vector3 position;
    math::Quat rotation;
    math::Diag3x3 scale;
    int32_t objectID;
    int32_t worldID;
};

struct BatchRenderable {
    Entity renderEntity;
};

struct BatchRenderCameraEntity {
    Entity cameraEntity;
    math::Vector3 cameraOffset;
};

struct BatchRenderInstance : public Archetype<
    InstanceData
> {};

struct BatchRenderCameraInstance : public Archetype<
    PerspectiveCameraData
> {};

struct BatchRendererECSBridge {
    PerspectiveCameraData *views;
    InstanceData *instances;
    // Each offset describes where the instances of a world are
    int32_t *worldInstanceOffsets;

    // We would also need an array of offsets. One offset for each world which
    // would lead us to first InstanceData of that world.

#if defined(MADRONA_BATCHRENDER_VK_RASTER)
#elif defined(MADRONA_BATCHRENDER_RT)
    AccelStructInstance *tlasInstancesBase;
    AccelStructRangeInfo *numInstances;
    uint64_t *blases;
    PackedViewData **packedViews;
    uint32_t *numInstancesReadback;
#elif defined(MADRONA_BATCHRENDER_METAL)
    PerspectiveCameraData **views;
    uint32_t *numViews;
    InstanceData *instanceData;
    uint32_t *numInstances;
#endif
    int32_t renderWidth;
    int32_t renderHeight;
    uint32_t maxViewsPerWorld;
};

struct BatchRenderingSystem {
    static constexpr uint32_t invalidViewIDX = 0xFFFF'FFFF;

    static void registerTypes(ECSRegistry &registry,
                              BatchRendererECSBridge *bridge);

    static TaskGraphNodeID setupTasks(TaskGraphBuilder &builder,
                                      Span<const TaskGraphNodeID> deps);

    static void reset(Context &ctx);

    static void init(Context &ctx,
                     const BatchRendererECSBridge *bridge);

    static void makeEntityRenderable(Context &ctx,
                                     Entity e);

    // Will make the entity a camera - this means that a view will correspond
    // to the rendered output of what the entity sees.
    static void attachEntityToView(Context &ctx,
                                   Entity e,
                                   float vfov_degrees,
                                   float z_near,
                                   int32_t view_idx,
                                   const math::Vector3 &camera_offset);
};

}
