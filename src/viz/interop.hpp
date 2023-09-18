#pragma once

namespace madrona::viz {

struct alignas(16) PerspectiveCameraData {
    math::Vector3 position;
    math::Quat rotation;
    float xScale;
    float yScale;
    float zNear;
    uint32_t pad[2];
};

struct alignas(16) InstanceData {
    math::Vector3 position;
    math::Quat rotation;
    math::Diag3x3 scale;
    int32_t objectID;
    uint32_t pad;
};

struct VizECSBridge {
    PerspectiveCameraData **views;
    uint32_t *numViews;
    InstanceData **instances;
    uint32_t *numInstances;
    int32_t renderWidth;
    int32_t renderHeight;
    bool *episodeDone;
    uint32_t *voxels;
};

}
