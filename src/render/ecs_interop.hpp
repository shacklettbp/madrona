#pragma once

#include <madrona/render/ecs.hpp>

namespace madrona::render {

struct RenderECSBridge {
    // Allocated from Vulkan, to be imported into Cuda
    PerspectiveCameraData *views;
    InstanceData *instances;
    TLBVHNode *aabbs;
    LightDesc *lights;

    int32_t *instanceOffsets;
    int32_t *viewOffsets;
    int32_t *lightOffsets;
    uint32_t *totalNumViews;
    uint32_t *totalNumInstances;
    uint32_t *totalNumLights;
    AtomicU32 *totalNumViewsCPUInc;
    AtomicU32 *totalNumInstancesCPUInc;
    AtomicU32 *totalNumLightsCPUInc;
    // Keys used for sorting (most significant 32 bits: world ID; 
    //                        least significant 32 bits: entity ID)
    uint64_t *instancesWorldIDs;
    uint64_t *viewsWorldIDs;
    uint64_t *lightWorldIDs;
    int32_t renderWidth;
    int32_t renderHeight;
    uint32_t *voxels;

    uint32_t maxViewsPerworld;
    uint32_t maxInstancesPerWorld;
    uint32_t maxLightsPerWorld;
    bool isGPUBackend;
};

}
