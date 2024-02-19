#pragma once

#include <madrona/render/ecs.hpp>

namespace madrona {

// Negative node indices are for leaves
struct LBVHNode {
    int32_t left;
    int32_t right;
};

// Internal data for the BVH
struct BVHInternalData {
    // These are the internal nodes. Needs to be properly allocated to
    // accomodate for the number of instances.
    LBVHNode *internalNodes;
    uint32_t numAllocatedNodes;

    AtomicU32 buildFastAccumulator;
};

struct BVHParams {
    uint32_t numWorlds;
    render::InstanceData *instances;
    render::PerspectiveCameraData *views;
    int32_t *instanceOffsets;
    int32_t *instanceCounts;
    int32_t *viewOffsets;
    uint32_t *mortonCodes;
    BVHInternalData *internalData;

    // These are all going to be inherited from the ECS
    void *hostAllocator;
    void *tmpAllocator;
};

}
