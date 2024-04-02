#pragma once

#include <madrona/mesh_bvh.hpp>
#include <madrona/dyn_array.hpp>
#include <madrona/math.hpp>
#include <madrona/span.hpp>
#include <madrona/optional.hpp>

namespace madrona {
namespace imp {

struct SourceMesh {
    math::Vector3 *positions;
    math::Vector3 *normals;
    math::Vector4 *tangentAndSigns;
    math::Vector2 *uvs;

    uint32_t *indices;
    uint32_t *faceCounts;
    uint32_t *faceMaterials;

    uint32_t numVertices;
    uint32_t numFaces;
    uint32_t materialIDX;
};

struct SourceObject {
    Span<SourceMesh> meshes;
    uint32_t bvhIndex;
};

struct SourceTexture {
    const char *path;
};

struct SourceMaterial {
    math::Vector4 color;

    // If this is -1, no texture will be applied. Otherwise,
    // the color gets multipled by color of the texture read in
    // at the UVs of the pixel.
    int32_t textureIdx;

    float roughness;
    float metalness;
};

struct SourceInstance {
    math::Vector3 translation;
    math::Quat rotation;
    math::Diag3x3 scale;
    uint32_t objIDX;
};

struct ImportedAssets {
    struct GPUGeometryData {
        render::MeshBVH::Node *nodes;
        uint32_t numNodes;

        render::MeshBVH::LeafGeometry *leafGeos;
        uint32_t numLeaves;

        math::Vector3 *vertices;
        uint32_t numVerts;

        render::MeshBVH *meshBVHs;
        uint32_t numBVHs;
    };

    struct GeometryData {
        DynArray<DynArray<math::Vector3>> positionArrays;
        DynArray<DynArray<math::Vector3>> normalArrays;
        DynArray<DynArray<math::Vector4>> tangentAndSignArrays;
        DynArray<DynArray<math::Vector2>> uvArrays;
        DynArray<DynArray<uint32_t>> indexArrays;
        DynArray<DynArray<uint32_t>> faceCountArrays;
        DynArray<DynArray<SourceMesh>> meshArrays;
        DynArray<DynArray<render::MeshBVH>> meshBVHArrays;
    } geoData;

    DynArray<SourceObject> objects;
    DynArray<SourceMaterial> materials;
    DynArray<SourceInstance> instances;

    static Optional<ImportedAssets> importFromDisk(
        Span<const char * const> asset_paths,
        Span<char> err_buf = { nullptr, 0 },
        bool one_object_per_asset = false,
        bool generate_mesh_bvhs = false);

    // Unfinished but provides just enough to support BVH raytracing.
    static Optional<GPUGeometryData> makeGPUData(
        const ImportedAssets &assets);
};

}
}
