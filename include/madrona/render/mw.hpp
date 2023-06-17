#pragma once

#include <madrona/math.hpp>
#include <madrona/taskgraph.hpp>

namespace madrona {
namespace render {

enum class CameraMode : uint32_t {
    Perspective,
    Lidar,
    None,
};

struct ViewID {
    int32_t idx;
};

struct ViewSettings {
    float xScale;
    float yScale;
    float zNear;
    math::Vector3 cameraOffset;
    ViewID viewID;
};

// FIXME: In the current design, the renderer is created before the
// ECS and the below struct provides the pointers necessary for the ECS
// to copy its data into the renderer. In a perfect world, it would be nice
// if the renderer could use the ECS and the below RenderingSystem could
// interface directly with the rendering APIs. You could then imagine a
// different RenderingSystem for each possible renderer implement
// (batch, viewer, etc)
struct RendererBridge;

struct RenderingSystem {
    static void registerTypes(ECSRegistry &registry);

    static TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                                        Span<const TaskGraph::NodeID> deps);

    static void reset(Context &ctx);

    static ViewSettings setupView(Context &ctx,
                                  float vfov_degrees,
                                  float z_near,
                                  math::Vector3 camera_offset,
                                  ViewID view_id);
};

}
}
