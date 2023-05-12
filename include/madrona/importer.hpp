#pragma once

#include <madrona/dyn_array.hpp>
#include <madrona/math.hpp>
#include <madrona/span.hpp>
#include <madrona/optional.hpp>

#include <string_view>

namespace madrona {
namespace imp {

struct SourceMesh {
    const math::Vector3 *positions;
    const math::Vector3 *normals;
    const math::Vector4 *tangentAndSigns;
    const math::Vector2 *uvs;
    const uint32_t *indices;
    const uint32_t *faceCounts;
    uint32_t numVertices;
    uint32_t numFaces;
    uint32_t materialIDX;
};

struct SourceObject {
    Span<const SourceMesh> meshes;
};

struct SourceMaterial {};

struct SourceInstance {
    math::Vector3 translation;
    math::Quat rotation;
    math::Diag3x3 scale;
    uint32_t objIDX;
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

    static Optional<ImportedAssets> importFromDisk(
        Span<const char * const> asset_paths,
        Span<char> err_buf = { nullptr, 0 },
        bool one_object_per_asset = false);
};

}
}
