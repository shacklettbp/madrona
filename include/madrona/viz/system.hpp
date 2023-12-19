#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>

namespace madrona::viz {

struct VizCamera {
    float xScale;
    float yScale;
    float zNear;
    math::Vector3 cameraOffset;
    int32_t viewIDX;
};

struct VizECSBridge;

struct VizRenderingSystem {
    static void registerTypes(ECSRegistry &registry);

    static TaskGraphNodeID setupTasks(
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> deps);

    static void reset(Context &ctx);

    static void init(Context &ctx,
                     const VizECSBridge *bridge);

    static VizCamera setupView(Context &ctx,
                               float vfov_degrees,
                               float z_near,
                               math::Vector3 camera_offset,
                               int32_t view_idx);

    static uint32_t * getVoxelPtr(Context &ctx);

    static void markEpisode(Context &ctx);
};

}
