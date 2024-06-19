#pragma once

#include <string>
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
};

enum class TextureLoadInfo {
    FileName,
    PixelBuffer,
};

enum class TextureFormat : int {
    KTX2,
    PNG,
    JPG,
    BC7,
};

struct BackingImageData {
    DynArray<uint8_t> imageData;
    size_t imageSize;
    TextureFormat format;
    uint32_t width = 0;
    uint32_t height = 0;
    bool processed = false;
};

struct PixelBufferInfo {
    uint32_t backingDataIndex;
    BackingImageData data;
};

struct SourceTextureConfig {
    size_t imageSize;
    TextureFormat format;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct SourceTexture {
    void *imageData;
    SourceTextureConfig config;
    uint32_t dataBufferIndex;
    inline SourceTexture();
    inline SourceTexture(const char *path_ptr);
    inline SourceTexture(TextureLoadInfo tex_info, const char *path_ptr);
    inline SourceTexture(PixelBufferInfo p_info);
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

struct SourceAssetInfo {
    uint32_t numObjects;
    std::string path;
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

    struct ImageData {
        DynArray<BackingImageData> imageArrays;
    } imgData;

    DynArray<SourceObject> objects;
    DynArray<SourceMaterial> materials;
    DynArray<SourceInstance> instances;
    DynArray<SourceTexture> texture;
    DynArray<SourceAssetInfo> assetInfos;

    static Optional<ImportedAssets> importFromDisk(
        Span<const char * const> asset_paths,
        Span<char> err_buf = { nullptr, 0 },
        bool one_object_per_asset = false);

    struct ProcessOutput {
        bool shouldCache;
        void *outputData;
        SourceTextureConfig newTex;
    };

    using TextureProcessFunc = ProcessOutput (*)(SourceTexture&);
    void postProcessTextures(const char *texture_cache, 
            TextureProcessFunc process_tex_func);
};

}
}

#include "importer.inl"
