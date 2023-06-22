#pragma once

#include <madrona/math.hpp>
#include <madrona/taskgraph.hpp>
#include <madrona/render/components.hpp>

namespace madrona {
namespace render {

struct BatchRendererBridge;

struct BatchRenderingSystem {
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
