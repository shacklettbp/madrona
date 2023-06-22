#pragma once

#include <madrona/state.hpp>
#include <madrona/taskgraph.hpp>
#include <madrona/render/components.hpp>

namespace madrona::viz {

struct ViewerECSBridge;

class ViewerRenderingSystem {
    static void registerTypes(ECSRegistry &registry);

    static TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                                        Span<const TaskGraph::NodeID> deps);

    static void reset(Context &ctx);

    static void init(Context &ctx,
                     const ViewerECSBridge *bridge);

};

}
