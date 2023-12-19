#pragma once

#include <madrona/math.hpp>
#include <madrona/taskgraph_builder.hpp>

namespace madrona::render {

struct BatchRenderCamera {
    float xScale;
    float yScale;
    float zNear;
    math::Vector3 cameraOffset;
    int32_t viewIDX;
};

struct BatchRendererECSBridge;

struct BatchRenderingSystem {
    static void registerTypes(ECSRegistry &registry);

    static TaskGraphNodeID setupTasks(TaskGraphBuilder &builder,
                                        Span<const TaskGraphNodeID> deps);

    static void reset(Context &ctx);

    static void init(Context &ctx,
                     const BatchRendererECSBridge *bridge);

    static BatchRenderCamera setupView(Context &ctx,
                                       float vfov_degrees,
                                       float z_near,
                                       math::Vector3 camera_offset,
                                       int32_t view_idx);
};

}
