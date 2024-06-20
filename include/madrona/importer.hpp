#pragma once

#include <string>
#include <madrona/mesh_bvh.hpp>
#include <madrona/dyn_array.hpp>
#include <madrona/math.hpp>
#include <madrona/span.hpp>
#include <madrona/optional.hpp>
#include <madrona/stack_alloc.hpp>

namespace madrona::imp {

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
};

enum class SourceTextureFormat : int32_t {
    R8G8B8A8,
    BC7,
};

struct SourceTexture {
    void *data;
    SourceTextureFormat format;
    uint32_t width;
    uint32_t height;
    size_t numBytes;
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

class ImageImporter {
public:
    ImageImporter();
    ImageImporter(ImageImporter &&);
    ~ImageImporter();

    using ImportHandler =
        Optional<SourceTexture> (*)(void *data, size_t num_bytes);

    int32_t addHandler(const char *extension, ImportHandler fn);

    int32_t getPNGTypeCode();
    int32_t getJPGTypeCode();
    int32_t getExtensionTypeCode(const char *extension);

    Optional<SourceTexture> importImage(
        void *data, size_t num_bytes, int32_t type_code);

    Optional<SourceTexture> importImage(const char *path);

    Span<SourceTexture> importImages(
        StackAlloc &tmp_alloc, Span<const char * const> paths);

    void deallocImportedImages(Span<SourceTexture> textures);
    

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct ImportedAssets {
    struct GeometryData {
        DynArray<DynArray<math::Vector3>> positionArrays;
        DynArray<DynArray<math::Vector3>> normalArrays;
        DynArray<DynArray<math::Vector4>> tangentAndSignArrays;
        DynArray<DynArray<math::Vector2>> uvArrays;

        DynArray<DynArray<uint32_t>> indexArrays;
        DynArray<DynArray<uint32_t>> faceCountArrays;

        DynArray<DynArray<SourceMesh>> meshArrays;
    } geoData;

    DynArray<SourceObject> objects;
    DynArray<SourceMaterial> materials;
    DynArray<SourceInstance> instances;
    DynArray<SourceTexture> textures;

};

class AssetImporter {
public:
    AssetImporter();
    AssetImporter(ImageImporter &&img_importer);
    AssetImporter(AssetImporter &&);
    ~AssetImporter();

    ImageImporter & imageImporter();

    Optional<ImportedAssets> importFromDisk(
        Span<const char * const> asset_paths,
        Span<char> err_buf = { nullptr, 0 },
        bool one_object_per_asset = false);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};


}
