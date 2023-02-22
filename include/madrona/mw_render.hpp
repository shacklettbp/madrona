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

struct RenderingSystem {
    static void registerTypes(ECSRegistry &registry);

    static TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                                        Span<const TaskGraph::NodeID> deps);

    static void reset(Context &ctx);

    static ViewSettings setupView(Context &ctx,
                                  float vfov_degrees,
                                  float aspect_ratio,
                                  float z_near,
                                  math::Vector3 camera_offset,
                                  ViewID view_id);
};

}
}
