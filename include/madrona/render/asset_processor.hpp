#pragma once

#include <madrona/importer.hpp>
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
    cudaTextureObject_t *textures;
    cudaArray_t *textureBuffers;
    Material *materials;
};

namespace AssetProcessor {
    Optional<MeshBVHData> makeBVHData(
            const imp::ImportedAssets &assets);

    MaterialData initMaterialData(
        const imp::SourceMaterial *materials,
        uint32_t num_materials,
        const imp::SourceTexture *textures,
        uint32_t num_textures);
};
    
}
