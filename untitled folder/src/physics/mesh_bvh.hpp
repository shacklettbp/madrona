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

    math::Vector3 *vertices;

    Node *nodes;
    LeafGeo *leafGeos;
    LeafMaterial *leafMaterials;

    uint32_t numNodes;
    uint32_t numLeaves;
    uint32_t numVerts;
};

}
