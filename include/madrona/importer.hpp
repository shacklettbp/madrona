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
    uint32_t *vertexMaterials;

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

enum TextureLoadInfo {
    FILE_NAME,
    PIXEL_BUFFER
};

enum TextureFormat {
    KTX2,
    PNG,
    JPG,
};

struct PixelBufferInfo {
    void *pixels;
    TextureFormat format;
    int bufferSize;
};

struct SourceTexture {
    TextureLoadInfo info;
    union {
        const char *path;
        PixelBufferInfo pix_info;
    };
    SourceTexture(const char *path_ptr) : info(FILE_NAME), path(path_ptr){
    }
    SourceTexture(TextureLoadInfo tex_info, const char *path_ptr) {
        info = tex_info;
        path = path_ptr;
    }
    SourceTexture(PixelBufferInfo p_info) {
        info = PIXEL_BUFFER;
        pix_info = p_info;
    }
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

struct EmbreeLoader {
    struct Impl;

    EmbreeLoader(Span<char> err_buf) {}
    EmbreeLoader(EmbreeLoader &&) = default;
    ~EmbreeLoader() {}

    // std::unique_ptr<Impl> impl_;

    Optional<render::MeshBVH> load(const SourceObject &obj, const DynArray<SourceMaterial> &materials) {}
};

struct ImportedAssets {
    struct GPUGeometryData {
        render::MeshBVH::Node *nodes;
        uint64_t numNodes;

        render::MeshBVH::LeafGeometry *leafGeos;
        render::MeshBVH::LeafMaterial *leafMaterial;
        uint64_t numLeaves;

        render::MeshBVH::BVHVertex *vertices;
        uint64_t numVerts;

        render::MeshBVH *meshBVHs;
        uint64_t numBVHs;
    };

    struct GeometryData {
        DynArray<DynArray<math::Vector3>> positionArrays;
        DynArray<DynArray<math::Vector3>> normalArrays;
        DynArray<DynArray<math::Vector4>> tangentAndSignArrays;
        DynArray<DynArray<math::Vector2>> uvArrays;

        // This is bad but we're currently assigning materials to vertices
        DynArray<DynArray<uint32_t>> materialIndices;

        DynArray<DynArray<uint32_t>> indexArrays;
        DynArray<DynArray<uint32_t>> faceCountArrays;

        DynArray<DynArray<SourceMesh>> meshArrays;
        DynArray<DynArray<render::MeshBVH>> meshBVHArrays;
    } geoData;

    struct ImageData{
        DynArray<DynArray<uint8_t>> imageArrays;
    } imgData;

    DynArray<SourceObject> objects;
    DynArray<SourceMaterial> materials;
    DynArray<SourceInstance> instances;
    DynArray<SourceTexture> texture;

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
