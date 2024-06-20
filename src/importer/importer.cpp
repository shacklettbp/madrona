#include <madrona/importer.hpp>

#include <madrona/dyn_array.hpp>
#include <madrona/heap_array.hpp>

#include <string_view>
#include <filesystem>
#include <string>

#include <meshoptimizer.h>

#include "obj.hpp"

#ifdef MADRONA_GLTF_SUPPORT
#include "gltf.hpp"
#endif

#ifdef MADRONA_USD_SUPPORT
#include "usd.hpp"
#endif

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

namespace madrona::imp {

using namespace math;

struct AssetImporter::Impl {
    ImageImporter imgImporter;

    Optional<OBJLoader> objLoader;

#ifdef MADRONA_GLTF_SUPPORT
    Optional<GLTFLoader> gltfLoader;
#endif

#ifdef MADRONA_USD_SUPPORT
    Optional<USDLoader> usdLoader;
#endif

    static inline Impl * make(ImageImporter &&img_importer);

    inline Optional<ImportedAssets> importFromDisk(
        Span<const char * const> asset_paths,
        Span<char> err_buf, bool one_object_per_asset);
};

AssetImporter::Impl * AssetImporter::Impl::make(ImageImporter &&img_importer)
{
    return new Impl {
        .imgImporter = std::move(img_importer),
        .objLoader = Optional<OBJLoader>::none(),
#ifdef MADRONA_GLTF_SUPPORT
        .gltfLoader = Optional<GLTFLoader>::none(),
#endif
#ifdef MADRONA_USD_SUPPORT
        .usdLoader = Optional<USDLoader>::none(),
#endif
    };
}

ImageImporter & AssetImporter::imageImporter()
{
    return impl_->imgImporter;
}

Optional<ImportedAssets> AssetImporter::Impl::importFromDisk(
    Span<const char * const> asset_paths,
    Span<char> err_buf, bool one_object_per_asset)
{
    ImportedAssets imported {
        .geoData = ImportedAssets::GeometryData {
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
        .textures { 0 },
    };

    bool load_success = false;
    for (const char *path : asset_paths) {
        std::string_view path_view(path);

        auto extension_pos = path_view.rfind('.');
        if (extension_pos == path_view.npos) {
            return Optional<ImportedAssets>::none();
        }
        auto extension = path_view.substr(extension_pos + 1);

        if (extension == "obj") {
            if (!objLoader.has_value()) {
                objLoader.emplace(err_buf);
            }

            load_success = objLoader->load(path, imported);
        } else if (extension == "gltf" || extension == "glb") {
#ifdef MADRONA_GLTF_SUPPORT
            if (!gltfLoader.has_value()) {
                gltfLoader.emplace(imgImporter, err_buf);
            }

            load_success = gltfLoader->load(
                path, imported, one_object_per_asset, imgImporter);
#else
            load_success = false;
            snprintf(err_buf.data(), err_buf.size(),
                     "Madrona not compiled with glTF support");
#endif
        } else if (extension == "usd" ||
                   extension == "usda" ||
                   extension == "usdc" ||
                   extension == "usdz") {
#ifdef MADRONA_USD_SUPPORT
            if (!usdLoader.has_value()) {
                usdLoader.emplace(imgImporter, err_buf);
            }

            load_success = usdLoader->load(
                path, imported, one_object_per_asset, imgImporter);
#else
            load_success = false;
            snprintf(err_buf.data(), err_buf.size(),
                     "Madrona not compiled with USD support");
#endif
        }

        if (!load_success) {
            printf("Load failed\n");
            break;
        }
    }

    if (!load_success) {
        return Optional<ImportedAssets>::none();
    }

    return imported;
}

AssetImporter::AssetImporter()
    : AssetImporter(ImageImporter())
{}

AssetImporter::AssetImporter(ImageImporter &&img_importer)
    : impl_(Impl::make(std::move(img_importer)))
{}

AssetImporter::AssetImporter(AssetImporter &&) = default;
AssetImporter::~AssetImporter() = default;

Optional<ImportedAssets> AssetImporter::importFromDisk(
    Span<const char * const> paths, Span<char> err_buf,
    bool one_object_per_asset)
{
    return impl_->importFromDisk(paths, err_buf, one_object_per_asset);
}

}
