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

// Make sure that this becomes packed in the future.
struct alignas(16) PerspectiveCameraDataBR {
    math::Vector4 position;
    math::Quat rotation;
    float xScale;
    float yScale;
    float zNear;
    int32_t viewIDX;
    int32_t worldIDX;
};

// Instance data that is needed to render an object
struct alignas(16) InstanceDataBR {
    math::Vector3 position;
    math::Quat rotation;
    math::Diag3x3 scale;
    int32_t objectID;
    int32_t worldID;
};

struct BatchRendererECSBridge {
    PerspectiveCameraDataBR *views;
    InstanceDataBR *instances;
    // Each offset describes where the instances of a world are
    int32_t *worldInstanceOffsets;
    int32_t renderWidth;
    int32_t renderHeight;
    uint32_t maxViewsPerWorld;
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
    BatchRendererECSBridge brBridge;
};

}
