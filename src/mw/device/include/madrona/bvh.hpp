#pragma once

#include <madrona/math.hpp>
#include <madrona/render/ecs.hpp>
#include <madrona/components.hpp>
#include <madrona/mesh_bvh.hpp>

#ifndef MADRONA_TLAS_WIDTH
#define MADRONA_TLAS_WIDTH 8
#endif

namespace madrona {

// This is the structure of the node which is just used for traversal
template <typename NodeIndex, int Width>
struct BVHNodeQuantized {
    using BVHNodeT = BVHNodeQuantized<NodeIndex, Width>;
    using NodeIndexT = NodeIndex;
    
    static constexpr int NodeWidth = Width;

    math::Vector3 minPoint;

    int8_t expX, expY, expZ;
    uint8_t numChildren;

    // Quantized min and max coordinates of the children
    uint8_t qMinX[Width], qMinY[Width], qMinZ[Width];
    uint8_t qMaxX[Width], qMaxY[Width], qMaxZ[Width];

    // We don't need to store the parent.
    // If the child is an internal node, it will be positive.
    // If the child is a leaf node, it will be negative and store
    // the instance index.
    //
    // NOTE: 0 is reserved for invalid nodes and also removes the ambiguity
    // between 0th leaf node and 0th internal node.
    NodeIndex childrenIdx[Width];

    math::AABB convertToAABB(uint32_t child_idx)
    {
        auto to_float = [](uint32_t data) -> float {
            return *((float *)&data);
        };

        return math::AABB {
            .pMin = {
                minPoint.x + to_float((expX + 127) << 23) * qMinX[child_idx],
                minPoint.y + to_float((expY + 127) << 23) * qMinY[child_idx],
                minPoint.z + to_float((expZ + 127) << 23) * qMinZ[child_idx],
            },

            .pMax = {
                minPoint.x + to_float((expX + 127) << 23) * qMaxX[child_idx],
                minPoint.y + to_float((expY + 127) << 23) * qMaxY[child_idx],
                minPoint.z + to_float((expZ + 127) << 23) * qMaxZ[child_idx],
            },
        };
    }

    static BVHNodeT construct(uint32_t num_children,
                              math::AABB *child_aabbs,
                              NodeIndex *child_indices)
    {
        assert(num_children <= Width);

        math::Vector3 root_min = child_aabbs[0].pMin,
                      root_max = child_aabbs[0].pMax;

        // Get the bounds of the parent of the given children
        for (uint32_t i = 1; i < num_children; ++i) {
            math::AABB &aabb = child_aabbs[i];

            root_min.x = fminf(root_min.x, aabb.pMin.x);
            root_min.y = fminf(root_min.y, aabb.pMin.y);
            root_min.z = fminf(root_min.z, aabb.pMin.z);

            root_max.x = fmaxf(root_max.x, aabb.pMax.x);
            root_max.y = fmaxf(root_max.y, aabb.pMax.y);
            root_max.z = fmaxf(root_max.z, aabb.pMax.z);
        }

        math::Vector3 root_extent = root_max - root_min;

        BVHNodeT ret = {
            .minPoint = root_min,
            .expX = (int8_t)ceilf(log2f(root_extent.x / (powf(2.f, 8.f) - 1.f))),
            .expY = (int8_t)ceilf(log2f(root_extent.y / (powf(2.f, 8.f) - 1.f))),
            .expZ = (int8_t)ceilf(log2f(root_extent.z / (powf(2.f, 8.f) - 1.f))),
            .numChildren = (uint8_t)num_children
        };

        for (int i = 0; i < num_children; ++i) {
            // Quantize the AABB of the child
            math::AABB &aabb = child_aabbs[i];

            ret.qMinX[i] = floorf((aabb.pMin.x - root_min.x) / powf(2, ret.expX));
            ret.qMinY[i] = floorf((aabb.pMin.y - root_min.y) / powf(2, ret.expY));
            ret.qMinZ[i] = floorf((aabb.pMin.z - root_min.z) / powf(2, ret.expZ));

            ret.qMaxX[i] = ceilf((aabb.pMax.x - root_min.x) / powf(2, ret.expX));
            ret.qMaxY[i] = ceilf((aabb.pMax.y - root_min.y) / powf(2, ret.expY));
            ret.qMaxZ[i] = ceilf((aabb.pMax.z - root_min.z) / powf(2, ret.expZ));

            ret.childrenIdx[i] = child_indices[i];
        }

        return ret;
    }
};

// The quantized BVH node used currently
using QBVHNode = BVHNodeQuantized<int16_t, MADRONA_TLAS_WIDTH>;

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
};

}
