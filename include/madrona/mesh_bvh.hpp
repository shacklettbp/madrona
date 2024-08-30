#pragma once

#include <madrona/types.hpp>
#include <madrona/geo.hpp>

#define MADRONA_COMPRESSED_DEINDEXED_TEX

#ifdef MADRONA_GPU_MODE
#include <madrona/mw_gpu/host_print.hpp>
#endif

#ifndef MADRONA_BLAS_WIDTH
#define MADRONA_BLAS_WIDTH 4
#endif

#ifndef MADRONA_BLAS_LEAF_WIDTH
#define MADRONA_BLAS_LEAF_WIDTH 2
#endif

namespace madrona {

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

struct TraversalStack {
    static constexpr CountT stackSize = 32;

    int32_t s[stackSize];
    uint32_t size;

    // if def for the shared version
    void push(int32_t v)
    {
        s[size++] = v;
    }

    int32_t pop()
    {
        return s[--size];
    }
};

struct MeshBVH {
    static constexpr inline CountT numTrisPerLeaf = MADRONA_BLAS_LEAF_WIDTH;
    static constexpr inline CountT nodeWidth = MADRONA_BLAS_WIDTH;
    static constexpr inline int32_t sentinel = (int32_t)0xFFFF'FFFF;

    struct Node {
        float minX;
        float minY;
        float minZ;
        int8_t expX;
        int8_t expY;
        int8_t expZ;
        uint8_t internalNodes;
        uint8_t triSize[nodeWidth];
        uint8_t qMinX[nodeWidth];
        uint8_t qMinY[nodeWidth];
        uint8_t qMinZ[nodeWidth];
        uint8_t qMaxX[nodeWidth];
        uint8_t qMaxY[nodeWidth];
        uint8_t qMaxZ[nodeWidth];
        int32_t children[nodeWidth];
        int32_t parentID;

        inline bool isLeaf(madrona::CountT child) const;
        inline int32_t leafIDX(madrona::CountT child) const;

        inline void setLeaf(madrona::CountT child, int32_t idx);
        inline void setInternal(madrona::CountT child, int32_t internal_idx);
        inline bool hasChild(madrona::CountT child) const;
        inline void clearChild(madrona::CountT child);
    };

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
        int32_t objectIDX;
    };

    template <typename Fn>
    void findOverlaps(const math::AABB &aabb, Fn &&fn) const;

#if 0
    inline bool traceRay(math::Vector3 ray_o,
                         math::Vector3 ray_d,
                         float *out_hit_t,
                         math::Vector3 *out_hit_normal,
                         void* shared,
                         TraversalStack *stack,
                         float t_max = float(FLT_MAX)) const;
#endif

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

    Node *nodes;
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
