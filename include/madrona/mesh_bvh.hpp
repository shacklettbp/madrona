#pragma once

#include <madrona/types.hpp>
#include <madrona/geo.hpp>

#define MADRONA_COMPRESSED_DEINDEXED_TEX

#ifdef MADRONA_GPU_MODE
#include <madrona/mw_gpu/host_print.hpp>
#endif

#define MADRONA_BVH_WIDTH 4

#ifndef MADRONA_BLAS_LEAF_WIDTH
#define MADRONA_BLAS_LEAF_WIDTH 2
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

    // If this is a bottom level BVH node
    uint8_t triSize[Width];

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

    math::AABB convertToAABB(uint32_t child_idx) const
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
                              int32_t *child_indices,
                              uint32_t my_index)
    {
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

        int iter = 0;
        for (; iter < num_children; ++iter) {
            // Quantize the AABB of the child
            math::AABB &aabb = child_aabbs[iter];

            ret.qMinX[iter] = floorf((aabb.pMin.x - root_min.x) / powf(2, ret.expX));
            ret.qMinY[iter] = floorf((aabb.pMin.y - root_min.y) / powf(2, ret.expY));
            ret.qMinZ[iter] = floorf((aabb.pMin.z - root_min.z) / powf(2, ret.expZ));

            ret.qMaxX[iter] = ceilf((aabb.pMax.x - root_min.x) / powf(2, ret.expX));
            ret.qMaxY[iter] = ceilf((aabb.pMax.y - root_min.y) / powf(2, ret.expY));
            ret.qMaxZ[iter] = ceilf((aabb.pMax.z - root_min.z) / powf(2, ret.expZ));

            if (child_indices[iter] < 0) {
                ret.childrenIdx[iter] = (uint32_t)(-child_indices[iter] - 1) | 
                                     0x8000'0000;
            } else {
                ret.childrenIdx[iter] = child_indices[iter] - 1;
            }
        }

        for (; iter < MADRONA_BVH_WIDTH; ++iter) {
            // Set the remaining ones to invalid node indices.
            ret.childrenIdx[iter] = 0xFFFF'FFFF;
        }

        return ret;
    }

    bool hasChild(uint32_t i) const
    {
        return childrenIdx[i] != 0xFFFF'FFFF;
    }

    bool isLeaf(uint32_t i) const
    {
        return childrenIdx[i] & 0x8000'0000;
    }

    uint32_t leafIDX(uint32_t i) const
    {
        return childrenIdx[i] & ~0x8000'0000;
    }
};

// The quantized BVH node used currently
using QBVHNode = BVHNodeQuantized<uint32_t, MADRONA_BVH_WIDTH>;

struct Material {
    // For now, just a color
    math::Vector4 color;

    int32_t textureIdx;

    float roughness;
    float metalness;
};

struct TriangleIndices {
    uint32_t indices[3];
};

struct MeshBVH {
    static constexpr inline CountT numTrisPerLeaf = MADRONA_BLAS_LEAF_WIDTH;
    static constexpr inline CountT nodeWidth = MADRONA_BVH_WIDTH;
    static constexpr inline int32_t sentinel = (int32_t)0xFFFF'FFFF;

    struct BVHMaterial{
        int32_t matIDX;
    };

    struct LeafMaterial {
        BVHMaterial material[1];
    };

    struct BVHVertex{
        madrona::math::Vector3 pos;
        madrona::math::Vector2 uv;
    };

    // Helper struct for Ray-Triangle intersection
    // following Woop et al 2013
    struct RayIsectTxfm {
        int32_t kx;
        int32_t ky;
        int32_t kz;
        float Sx;
        float Sy;
        float Sz;
        int32_t nearX;
        int32_t nearY;
        int32_t nearZ;
        int32_t farX;
        int32_t farY;
        int32_t farZ;
        math::Vector3 oNear;
        math::Vector3 oFar;
        math::Vector3 invDirNear;
        math::Vector3 invDirFar;
    };

    struct HitInfo {
        float tHit;
        math::Vector3 normal;
        math::Vector2 uv;

        MeshBVH *bvh;

        uint32_t leafMaterialIDX;
    };

    template <typename Fn>
    void findOverlaps(const math::AABB &aabb, Fn &&fn) const;

    inline bool traceRay(math::Vector3 ray_o,
                         math::Vector3 ray_d,
                         HitInfo *out_hit_info,
                         int32_t *stack,
                         int32_t &stack_size,
                         float t_max = float(FLT_MAX)) const;

    inline float sphereCast(math::Vector3 ray_o,
                            math::Vector3 ray_d,
                            float sphere_r,
                            math::Vector3 *out_hit_normal,
                            float t_max = float(FLT_MAX));

    inline bool traceRayLeaf(
        int32_t leaf_idx,
        int32_t num_tris,
        RayIsectTxfm tri_isect_txfm,
        math::Vector3 ray_o,
        float t_max,
        HitInfo *hit_info) const;

    inline bool traceRayLeafIndexed(int32_t leaf_idx,
                           int32_t i,
                           MeshBVH::RayIsectTxfm tri_isect_txfm,
                           math::Vector3 ray_o,
                           float t_max,
                           float *out_hit_t,
                           math::Vector3 *out_hit_normal) const;

    inline bool rayTriangleIntersection(
        math::Vector3 tri_a, math::Vector3 tri_b, math::Vector3 tri_c,
        int32_t kx, int32_t ky, int32_t kz,
        float Sx, float Sy, float Sz,
        math::Vector3 org,
        float t_max,
        float *out_hit_t,
        math::Vector3 *bary_out,
        math::Vector3 *out_hit_normal) const;

    inline bool fetchLeafTriangle(CountT leaf_idx,
                                  CountT offset,
                                  math::Vector3 *a,
                                  math::Vector3 *b,
                                  math::Vector3 *c,
                                  math::Vector2 *uv_a,
                                  math::Vector2 *uv_b,
                                  math::Vector2 *uv_c) const;

    static inline RayIsectTxfm computeRayIsectTxfm(
        math::Vector3 o, math::Vector3 d, math::Diag3x3 inv_d,
        math::AABB root_aabb);

    inline RayIsectTxfm computeRayIsectTxfm(
        math::Vector3 o, math::Vector3 d, math::Diag3x3 inv_d) const;

    inline bool sphereCastNodeCheck(math::Vector3 ray_o,
                                    math::Diag3x3 inv_d,
                                    float t_max,
                                    float sphere_r,
                                    math::AABB aabb) const;

    inline float sphereCastLeaf(int32_t leaf_idx,
                                math::Vector3 ray_o,
                                math::Vector3 ray_d,
                                float t_max,
                                float sphere_r,
                                math::Vector3 *out_hit_normal) const;

    inline float sphereCastTriangle(math::Vector3 tri_a,
                                    math::Vector3 tri_b,
                                    math::Vector3 tri_c,
                                    math::Vector3 ray_o,
                                    math::Vector3 ray_d,
                                    float t_max,
                                    float sphere_r,
                                    math::Vector3 *out_hit_normal) const;

    inline uint32_t getMaterialIDX(const HitInfo &info) const;
    inline uint32_t getMaterialIDX(int32_t mat_idx) const;

    QBVHNode *nodes;
    LeafMaterial *leafMats;

    BVHVertex *vertices;

    math::AABB rootAABB;
    uint32_t numNodes;
    uint32_t numLeaves;
    uint32_t numVerts;

    int32_t materialIDX;

    uint32_t magic;
};

}

#include "mesh_bvh.inl"
