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

#include <span>
#include <array>

using bytes = std::span<const std::byte>;

template <>
struct std::hash<bytes>
{
    std::size_t operator()(const bytes& x) const noexcept
    {
        return std::hash<std::string_view>{}({reinterpret_cast<const char*>(x.data()), x.size()});
    }
};

namespace madrona::imp {

using namespace math;

void ImportedAssets::postProcessTextures(const char *texture_cache, TextureProcessFunc process_tex_func)
{
    for (SourceTexture& tx: texture) {
        std::filesystem::path cache_dir = texture_cache;

        if (texture_cache) {
            cache_dir = texture_cache;
        }

        auto texture_hasher = std::hash<bytes>{};
        std::size_t hash = texture_hasher(
                std::as_bytes(
                        std::span((char *)tx.imageData,
                                  tx.config.imageSize)));

        std::string hash_str = std::to_string(hash);

        uint8_t *pixel_data = nullptr;
        uint32_t pixel_data_size = 0;

        if (texture_cache) {
            std::string path_to_cached_tex = (cache_dir / hash_str);

            FILE *read_fp = fopen(path_to_cached_tex.c_str(), "rb");

            if (read_fp) {
                fread(&tx.config, sizeof(SourceTextureConfig), 1, read_fp);

                pixel_data = (uint8_t *) malloc(tx.config.imageSize);

                imgData.imageArrays[tx.dataBufferIndex].imageData.clear();
                imgData.imageArrays[tx.dataBufferIndex].imageData.resize(tx.config.imageSize,[](uint8_t*){});
                fread(imgData.imageArrays[tx.dataBufferIndex].imageData.data(), tx.config.imageSize, 1, read_fp);
                tx.imageData = imgData.imageArrays[tx.dataBufferIndex].imageData.data();

                free(pixel_data);
                fclose(read_fp);
            } else {
                auto processOutput = process_tex_func(tx);

                if (!processOutput.shouldCache)
                    continue;

                tx.config = processOutput.newTex;
                imgData.imageArrays[tx.dataBufferIndex].imageData.clear();
                imgData.imageArrays[tx.dataBufferIndex].imageData.resize(tx.config.imageSize,[](uint8_t*){});
                memcpy(imgData.imageArrays[tx.dataBufferIndex].imageData.data(), processOutput.outputData,
                       tx.config.imageSize);
                tx.imageData = imgData.imageArrays[tx.dataBufferIndex].imageData.data();

                free(processOutput.outputData);

                pixel_data = imgData.imageArrays[tx.dataBufferIndex].imageData.data();
                pixel_data_size = tx.config.imageSize;

                // Write this data to the cache
                FILE *write_fp = fopen(path_to_cached_tex.c_str(), "wb");

                fwrite(&tx.config, sizeof(SourceTextureConfig), 1, write_fp);
                fwrite(pixel_data, pixel_data_size, 1, write_fp);

                fclose(write_fp);
            }
        } else {
            auto processOutput = process_tex_func(tx);

            if (!processOutput.shouldCache)
                continue;

            tx.config = processOutput.newTex;
            imgData.imageArrays[tx.dataBufferIndex].imageData.clear();
            imgData.imageArrays[tx.dataBufferIndex].imageData.resize(tx.config.imageSize,[](uint8_t*){});
            memcpy(imgData.imageArrays[tx.dataBufferIndex].imageData.data(), processOutput.outputData,
                       tx.config.imageSize);
            tx.imageData = imgData.imageArrays[tx.dataBufferIndex].imageData.data();

            free(processOutput.outputData);
        }
    }
}

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
        .imgData = ImageData {
            .imageArrays {0},
        },
        .objects { 0 },
        .materials { 0 },
        .instances { 0 },
        .texture = DynArray<SourceTexture>(0),
        .assetInfos { 0 },
    };

    auto obj_loader = Optional<OBJLoader>::none();
#ifdef MADRONA_GLTF_SUPPORT
    auto gltf_loader = Optional<GLTFLoader>::none();
#endif
#ifdef MADRONA_USD_SUPPORT
    auto usd_loader = Optional<USDLoader>::none();
#endif

    bool load_success = false;
    for (const char *path : paths) {
        uint32_t pre_objects_offset = imported.objects.size();

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
#ifdef MADRONA_GLTF_SUPPORT
            if (!gltf_loader.has_value()) {
                gltf_loader.emplace(err_buf);
            }

            load_success = gltf_loader->load(path, imported,
                                             one_object_per_asset);
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
            printf("Load failed\n");
            break;
        }

        uint32_t post_objects_offset = imported.objects.size();

        imported.assetInfos.push_back(SourceAssetInfo{
            post_objects_offset - pre_objects_offset,
            std::string(path_view)
        });
    }

    if (!load_success) {
        return Optional<ImportedAssets>::none();
    }

    return imported;
}

}
