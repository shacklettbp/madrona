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

struct ViewID {
    int32_t idx;
};

struct ViewSettings {
    float tanFOV;
    math::Vector3 cameraOffset;
    ViewID viewID;
};

// FIXME this is a copy of the PackedCamera / ViewData
// struct in render/vk/shaders/shader_common.h
struct PackedViewData {
    math::Quat rotation;
    math::Vector4 posAndTanFOV;
};

struct RendererInterface {
    AccelStructInstance **tlasInstancePtrs;
    uint32_t *tlasInstanceCounts;
    uint64_t *blases;
    PackedViewData **packedViews;
};

struct RendererInit {
    RendererInterface iface;
    math::Vector3 worldOffset;
};

struct RenderingSystem {
    static void registerTypes(ECSRegistry &registry);

    static TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                                        Span<const TaskGraph::NodeID> deps);

    static void init(Context &ctx, const RendererInit &init);

    static ViewSettings setupView(Context &ctx, float vfov_degrees,
                                  math::Vector3 camera_offset,
                                  ViewID view_id);
};

}
}
