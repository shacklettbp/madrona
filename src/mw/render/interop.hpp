#pragma once

#include <madrona/render/batch_renderer_system.hpp>

#if defined(MADRONA_LINUX) or defined(MADRONA_WINDOWS) or defined(MADRONA_GPU_MODE)
// #define MADRONA_BATCHRENDER_RT (1)
// #define MADRONA_BATCHRENDER_VK_RASTER (1)
#elif defined(MADRONA_MACOS)
#define MADRONA_BATCHRENDER_METAL (1)
#endif

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

#if defined(MADRONA_BATCHRENDER_METAL) || defined(MADRONA_BATCHRENDER_VK_RASTER)

#endif

struct BatchRendererState {
#if defined(MADRONA_BATCHRENDER_VK_RASTER)
    PerspectiveCameraData *views;
    uint32_t *numViews;
    InstanceData *instances;
#elif defined(MADRONA_BATCHRENDER_RT)
    AccelStructInstance *tlasInstanceBuffer;
    AccelStructRangeInfo *numInstances;
    uint64_t *blases;
    PackedViewData *packedViews;
#ifdef MADRONA_GPU_MODE
    uint32_t *count_readback;
#endif
#elif defined(MADRONA_BATCHRENDER_METAL)
    PerspectiveCameraData *views;
    uint32_t *numViews;
    InstanceData *instanceData;
    uint32_t *numInstances;
#endif
    int32_t renderWidth;
    int32_t renderHeight;
    float aspectRatio;
};

}
