#pragma once

#include <madrona/math.hpp>
#include <madrona/taskgraph.hpp>

namespace madrona {
namespace render {

struct AccelStructTransform {
    float matrix[3][4];
};

struct AccelStructInstance {
    AccelStructTransform transform;
    uint32_t instanceCustomIndex:24;
    uint32_t mask:8;
    uint32_t instanceShaderBindingTableRecordOffset:24;
    uint32_t flags:8;
    uint64_t accelerationStructureReference;
};

struct ViewSettings {
    float tanFOV;
    math::Vector3 cameraOffset;
};

struct ViewID {
    int32_t idx;
};

struct RenderingSystem {
    static void registerTypes(ECSRegistry &registry);

    static TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                                        Span<const TaskGraph::NodeID> deps);

    static void init(Context &ctx);

    static ViewSettings setupView(Context &ctx, float vfov_degrees,
                                  math::Vector3 camera_offset);
};

}
}
