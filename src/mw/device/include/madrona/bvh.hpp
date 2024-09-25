#pragma once

#include <madrona/math.hpp>
#include <madrona/render/ecs.hpp>
#include <madrona/components.hpp>
#include <madrona/mesh_bvh.hpp>

#ifndef MADRONA_TLAS_WIDTH
#define MADRONA_TLAS_WIDTH 8
#endif

namespace madrona {

// This isn't going to be the representation that actually gets traversed
// through. This is just for construction purposes.
struct LBVHNode {
    // The indices stored in here actually start at 1, not 0. This is because
    // 0 is reserved for invalid nodes.

    int32_t left;
    int32_t right;

    uint32_t instanceIdx;

    // If this is 0xFFFFFFFF, this is the root. Index starts at 0.
    uint32_t parent;
    math::AABB aabb;

    AtomicU32 reachedCount;

    // If both left and right are -1, then the node is invalid.
    // However, if just one of them is negative, then the negative index
    // is actually an index into the leaf nodes buffer.
    inline bool isInvalid()
    {
        return left == -1 && right == -1;
    }

    static inline int32_t childIdxToStoreIdx(int32_t idx, bool is_leaf)
    {
        if (is_leaf) {
            return -(idx + 1);
        } else {
            return idx + 1;
        }
    }

    static inline int32_t storeIdxToChildIdx(int32_t idx, bool &is_leaf)
    {
        if (idx < 0) {
            is_leaf = true;
            return (-idx) - 1;
        } else {
            is_leaf = false;
            return idx - 1;
        }
    }

    inline uint32_t numChildren()
    {
        return (uint32_t)(left != 0) + (uint32_t)(right != 0);
    }

    // TODO: IMPLEMENT SAH RIGHT NAOW!!!
    static inline float sah()
    {
        return 0.f;
    }
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

    // These are the final nodes
    QBVHNode *traversalNodes;

    AtomicU32 buildFastAccumulator;
    AtomicU32 buildSlowAccumulator;
    AtomicU32 optFastAccumulator;
    AtomicU32 constructAABBsAccumulator;

    uint32_t numFrames;

    AtomicU32 treeletRootIndexCounter;

    // NOTE: may use this for treelet optimization later
    uint32_t *indirectIndices;

    uint32_t numViews;
};

struct KernelTimingInfo {
    AtomicU32 timingCounts;
    AtomicU64 tlasTime;
    AtomicU64 blasTime;
    AtomicU64 numTLASTraces;
    AtomicU64 numBLASTraces;
    AtomicU64 memoryUsage;
};

struct BVHParams {
    uint32_t numWorlds;
    render::InstanceData *instances;
    render::PerspectiveCameraData *views;
    render::TLBVHNode *aabbs;
    int32_t *instanceOffsets;
    int32_t *instanceCounts;
    int32_t *viewOffsets;
    int32_t *viewCounts;
    uint32_t *mortonCodes;
    BVHInternalData *internalData;

    ::madrona::MeshBVH *bvhs;

    void *rgbOutput;
    void *depthOutput;
    uint32_t renderOutputResolution;

    // This will render the depth component as well as the RGB component.
    uint32_t raycastRGBD;

    // These are all going to be inherited from the ECS
    void *hostAllocator;
    void *tmpAllocator;
    void *hostPrintAddr;

    KernelTimingInfo *timingInfo;

    Material *materials;
    cudaTextureObject_t *textures;

    float nearSphere;

    // Used to determine how many thread blocks per SM.
    uint32_t numSMs;
    uint32_t smSharedMemory;
};

}
