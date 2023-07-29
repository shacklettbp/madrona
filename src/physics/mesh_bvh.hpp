#pragma once

namespace madrona::phys {

enum class MadronaCollisionFlags : uint64_t {
    BlocksRaycasts = 1 << 0,
};

struct MadronaCollisionMaterial {
    MadronaCollisionFlags flags;
};

struct MeshBVH {
    struct Node {
        float minX[4];
        float minY[4];
        float minZ[4];
        float maxX[4];
        float maxY[4];
        float maxZ[4];
        int32_t children[4];
        int32_t parentID;

        inline bool isLeaf(madrona::CountT child) const;
        inline int32_t leafIDX(madrona::CountT child) const;

        inline void setLeaf(madrona::CountT child, int32_t idx);
        inline void setInternal(madrona::CountT child, int32_t internal_idx);
        inline bool hasChild(madrona::CountT child) const;
        inline void clearChild(madrona::CountT child);
    };

    struct Leaf {
        uint32_t vertOffset;
        uint32_t triOffset;
        uint32_t numTris;
        uint32_t parentID;
    };

    Node *nodes;
    Leaf *leaves;

    madrona::math::Vector3 *vertices;
    uint32_t *indices;
    uint32_t *triCollisionMaterials;

    uint32_t numNodes;
    uint32_t numLeaves;
    uint32_t numVerts;
    uint32_t numTriangles;

    static constexpr int32_t sentinel = (int32_t)0xFFFF'FFFF;
};

}
