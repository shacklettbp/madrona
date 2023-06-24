#pragma once

#include <madrona/taskgraph.hpp>
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

    static TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                                        Span<const TaskGraph::NodeID> deps);

    static void reset(Context &ctx);

    static void init(Context &ctx,
                     const VizECSBridge *bridge);

    static VizCamera setupView(Context &ctx,
                               float vfov_degrees,
                               float z_near,
                               math::Vector3 camera_offset,
                               int32_t view_idx);

    static void markEpisode(Context &ctx);
};

}
