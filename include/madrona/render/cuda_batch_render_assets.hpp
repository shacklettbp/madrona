#pragma once

#include <madrona/mesh_bvh.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#else
using cudaTextureObject_t = uint32_t;
using cudaArray_t = uint32_t;
#endif

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
    cudaTextureObject_t *textures;
    cudaArray_t *textureBuffers;
    Material *materials;
};


}
