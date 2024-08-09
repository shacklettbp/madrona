#pragma once

#include <madrona/cuda_utils.hpp>
#include <madrona/mesh_bvh.hpp>

namespace madrona::render {

struct MeshBVHData {
    MeshBVH::Node *nodes;
    uint64_t numNodes;

    MeshBVH::LeafGeometry *leafGeos;
    MeshBVH::LeafMaterial *leafMaterial;
    uint64_t numLeaves;

    MeshBVH::BVHVertex *vertices;
    uint64_t numVerts;

    MeshBVH *meshBVHs;
    uint64_t numBVHs;
};

struct MaterialData {
    // GPU buffer containing array of texture objects
    void *textures;
    void *textureBuffers;
    Material *materials;
};


}
