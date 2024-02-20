#pragma once

#include <madrona/math.hpp>
#include <madrona/render/ecs.hpp>

namespace madrona {

#if 0
// Negative node indices are for leaves
struct LBVHNode {
    int32_t left;
    int32_t right;
    int32_t parent;
};

struct LeafNode {
    AABB aabb;
    int32_t parent;
};
#endif

// This isn't going to be the representation that actually gets traversed
// through. This is just for construction purposes.
struct LBVHNode {
    int32_t left;
    int32_t right;
    int32_t parent;
    math::AABB aabb;
};

// These have to be in global memory
struct TreeletFormationNode {
    // Total number of leaves under this treelet node.
    AtomicU32 numLeaves;

    // Number of threads which have reached this node.
    AtomicU32 numReached;
};

// Internal data for the BVH
struct BVHInternalData {
    // These are the internal nodes. Needs to be properly allocated to
    // accomodate for the number of instances.
    LBVHNode *internalNodes;
    TreeletFormationNode *treeletFormNodes;
    uint32_t numAllocatedNodes;

    LBVHNode *leaves;
    uint32_t numAllocatedLeaves;

    AtomicU32 buildFastAccumulator;
    AtomicU32 optFastAccumulator;
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
