#pragma once

namespace madrona::viz {

struct alignas(16) PerspectiveCameraData {
    math::Vector3 position;
    math::Quat rotation;
    float xScale;
    float yScale;
    float zNear;
    int32_t worldIDX;
    uint32_t pad;
};

struct alignas(16) InstanceData {
    math::Vector3 position;
    math::Quat rotation;
    math::Diag3x3 scale;
    int32_t objectID;
    int32_t worldIDX;
};

struct VizECSBridge {
    // Allocated from Vulkan, to be imported into Cuda
    PerspectiveCameraData *views;
    InstanceData *instances;

    int32_t *instanceOffsets;

    uint32_t *totalNumViews;
    uint32_t *totalNumInstances;

    AtomicU32 *totalNumViewsCPUInc;
    AtomicU32 *totalNumInstancesCPUInc;

    // Keys used for sorting (most significant 32 bits: world ID; 
    //                        least significant 32 bits: entity ID)
    uint64_t *instancesWorldIDs;
    uint64_t *viewsWorldIDs;

    int32_t renderWidth;
    int32_t renderHeight;
    bool *episodeDone;
    uint32_t *voxels;

    bool isGPUBackend;
};

}
