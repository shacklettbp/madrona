#pragma once

#include <madrona/math.hpp>
#include <madrona/taskgraph.hpp>

namespace madrona {
namespace render {

struct ObjectToWorld : math::Mat3x4 {
    ObjectToWorld(math::Mat3x4 mat)
        : Mat3x4(mat)
    {}
};

struct ObjectID {
    int32_t idx;
};

struct RenderObject : madrona::Archetype<ObjectToWorld, ObjectID> {};

struct RenderEntity {
    Entity renderEntity;
};

struct RenderSetupSystem {
    static void registerTypes(ECSRegistry &registry);

    static TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                                        Span<const TaskGraph::NodeID> deps);
};

}
}
