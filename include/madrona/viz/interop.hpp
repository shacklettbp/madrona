#pragma once

namespace madrona::viz {

struct alignas(16) PerspectiveCameraData {
    math::Vector4 position;
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
#if 0
    PerspectiveCameraData **views;
    uint32_t *numViews;
    InstanceData **instances;
    uint32_t *numInstances;
    int32_t renderWidth;
    int32_t renderHeight;
    bool *episodeDone;
    uint32_t *voxels;
#endif

    // Allocated from Vulkan, to be imported into Cuda
    PerspectiveCameraData *views;
    InstanceData *instances;
    int32_t *instanceOffsets;

    uint32_t *totalNumViews;

    int32_t renderWidth;
    int32_t renderHeight;
    bool *episodeDone;
    uint32_t *voxels;
};

}
