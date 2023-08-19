#pragma once

#include <madrona/types.hpp>

namespace madrona::phys {

enum class CollisionFlags : uint64_t {
    BlocksRaycasts = 1 << 0,
};

struct CollisionMaterial {
    CollisionFlags flags;
};

struct MeshBVH {
    static constexpr inline int32_t numTrisPerLeaf = 8;
    static constexpr inline int32_t nodeWidth = 4;
    static constexpr inline int32_t sentinel = (int32_t)0xFFFF'FFFF;

    struct Node {
        float minX[nodeWidth];
        float minY[nodeWidth];
        float minZ[nodeWidth];
        float maxX[nodeWidth];
        float maxY[nodeWidth];
        float maxZ[nodeWidth];
        int32_t children[nodeWidth];
        int32_t parentID;

        inline bool isLeaf(madrona::CountT child) const;
        inline int32_t leafIDX(madrona::CountT child) const;

        inline void setLeaf(madrona::CountT child, int32_t idx);
        inline void setInternal(madrona::CountT child, int32_t internal_idx);
        inline bool hasChild(madrona::CountT child) const;
        inline void clearChild(madrona::CountT child);
    };

    struct LeafGeometry {
        uint64_t packedIndices[numTrisPerLeaf];
    };

    struct LeafMaterial {
        uint32_t material[numTrisPerLeaf];
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

    template <typename Fn>
    void findOverlaps(const math::AABB &aabb, Fn &&fn) const;

    inline bool traceRay(math::Vector3 o,
                         math::Vector3 d,
                         float *out_hit_t,
                         math::Vector3 *out_hit_normal,
                         float t_max = float(FLT_MAX)) const;

    inline bool traceRayIntoLeaf(
        int32_t leaf_idx,
        RayIsectTxfm tri_isect_txfm,
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
        math::Vector3 *out_hit_normal) const;

    inline bool fetchLeafTriangle(CountT leaf_idx,
                                  CountT offset,
                                  math::Vector3 *a,
                                  math::Vector3 *b,
                                  math::Vector3 *c) const;

    inline RayIsectTxfm computeRayIsectTxfm(
        math::Vector3 o, math::Vector3 d, math::Diag3x3 inv_d) const;

    Node *nodes;
    LeafGeometry *leafGeos;
    LeafMaterial *leafMats;

    math::Vector3 *vertices;

    math::AABB rootAABB;
    uint32_t numNodes;
    uint32_t numLeaves;
    uint32_t numVerts;
};

}

#include "mesh_bvh.inl"
