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
    float tanFOV;
    math::Vector3 cameraOffset;
    ViewID viewID;
};

struct RenderingSystem {
    static void registerTypes(ECSRegistry &registry);

    static TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                                        Span<const TaskGraph::NodeID> deps);

    static ViewSettings setupView(Context &ctx, float vfov_degrees,
                                  math::Vector3 camera_offset,
                                  ViewID view_id);
};

}
}
