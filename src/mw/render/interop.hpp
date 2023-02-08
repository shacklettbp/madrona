#pragma once

#include <madrona/mw_render.hpp>

namespace madrona::render {
 
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

// FIXME this is a copy of the PackedCamera / ViewData
// struct in render/vk/shaders/shader_common.h
struct PackedViewData {
    math::Quat rotation;
    math::Vector4 posAndTanFOV;
};

// FIXME this is a copy of VkAccelerationStructureBuildRangeInfoKHR
struct AccelStructRangeInfo {
    uint32_t primitiveCount;
    uint32_t primitiveOffset;
    uint32_t firstVertex;
    uint32_t transformOffset;
};

struct RendererInterface {
    AccelStructInstance *tlasInstancesBase;
    AccelStructRangeInfo *numInstances;
    uint64_t *blases;
    PackedViewData **packedViews;
    uint32_t *numInstancesReadback;
};

struct RendererInit {
    RendererInterface iface;
    math::Vector3 worldOffset;
};

struct RendererState {
    AccelStructInstance *tlasInstanceBuffer;
    AccelStructRangeInfo *numInstances;
    uint64_t *blases;
    PackedViewData *packedViews;
    math::Vector3 worldOffset;
#ifdef MADRONA_GPU_MODE
    uint32_t *count_readback;
#endif

    static void init(Context &ctx,
                     const RendererInit &init);
};

}
