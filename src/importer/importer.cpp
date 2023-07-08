#include <madrona/importer.hpp>
#include <madrona/dyn_array.hpp>
#include <madrona/heap_array.hpp>

#include <string_view>

#include <meshoptimizer.h>

#include "obj.hpp"
#include "gltf.hpp"

#ifdef MADRONA_USD_SUPPORT
#include "usd.hpp"
#endif

namespace madrona::imp {

using namespace math;

Optional<ImportedAssets> ImportedAssets::importFromDisk(
    Span<const char * const> paths, Span<char> err_buf,
    bool one_object_per_asset)
{
    ImportedAssets imported {
        .geoData = GeometryData {
            .positionArrays { 0 },
            .normalArrays { 0 },
            .tangentAndSignArrays { 0 },
            .uvArrays { 0 },
            .indexArrays { 0 },
            .faceCountArrays { 0 },
            .meshArrays { 0 },
        },
        .objects { 0 },
        .materials { 0 },
        .instances { 0 },
    };

    auto obj_loader = Optional<OBJLoader>::none();
    auto gltf_loader = Optional<GLTFLoader>::none();
#ifdef MADRONA_USD_SUPPORT
    auto usd_loader = Optional<USDLoader>::none();
#endif

    bool load_success = false;
    for (const char *path : paths) {
        std::string_view path_view(path);

        auto extension_pos = path_view.rfind('.');
        if (extension_pos == path_view.npos) {
            return Optional<ImportedAssets>::none();
        }
        auto extension = path_view.substr(extension_pos + 1);

        if (extension == "obj") {
            if (!obj_loader.has_value()) {
                obj_loader.emplace(err_buf);
            }

            load_success = obj_loader->load(path, imported);
        } else if (extension == "gltf" || extension == "glb") {
            if (!gltf_loader.has_value()) {
                gltf_loader.emplace(err_buf);
            }

            load_success = gltf_loader->load(path, imported,
                                             one_object_per_asset);
        } else if (extension == "usd" ||
                   extension == "usda" ||
                   extension == "usdc" ||
                   extension == "usdz") {
#ifdef MADRONA_USD_SUPPORT
            if (!usd_loader.has_value()) {
                usd_loader.emplace(err_buf);
            }

            load_success = usd_loader->load(path, imported,
                                            one_object_per_asset);
#else
            load_success = false;
            snprintf(err_buf.data(), err_buf.size(),
                     "Madrona not compiled with USD support");
#endif
        }

        if (!load_success) {
            break;
        }
    }

    if (!load_success) {
        return Optional<ImportedAssets>::none();
    }

    return imported;
}

}
