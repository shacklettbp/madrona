#include <madrona/importer.hpp>

#include <madrona/dyn_array.hpp>
#include <madrona/heap_array.hpp>

#include <string_view>
#include <filesystem>
#include <string>

#include <meshoptimizer.h>

#include "obj.hpp"
#include "gltf.hpp"

#ifdef MADRONA_USD_SUPPORT
#include "usd.hpp"
#endif

#include "embree.hpp"

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

void ImportedAssets::postProcessTextures(const char *texture_cache, TextureProcessFunc process_tex_func) {
    printf("Processing Textures\n");
    for(SourceTexture& tx: texture) {
        tx.pix_info.data = imgData.imageArrays[tx.pix_info.backingDataIndex];
        if (tx.info == imp::TextureLoadInfo::PixelBuffer && !tx.pix_info.data.processed) {
            std::filesystem::path cache_dir = texture_cache;

            if (texture_cache) {
                cache_dir = texture_cache;
            }

            auto texture_hasher = std::hash < bytes > {};
            std::size_t hash = texture_hasher(
                    std::as_bytes(
                            std::span(tx.pix_info.data.imageData,
                                      tx.pix_info.data.imageSize)));

            std::string hash_str = std::to_string(hash);

            bool should_construct = false;

            uint8_t *pixel_data;
            uint32_t pixel_data_size = 0;

            uint32_t width, height;

            TextureFormat format;

            if (texture_cache) {
                std::string path_to_cached_tex = (cache_dir / hash_str);

                FILE *read_fp = fopen(path_to_cached_tex.c_str(), "rb");

                if (read_fp) {
                    printf("*");

                    BackingImageData data;
                    fread(&data, sizeof(BackingImageData), 1, read_fp);

                    pixel_data = (uint8_t *)malloc(data.imageSize);

                    fread(pixel_data, data.imageSize, 1, read_fp);
                    
                    data.imageData = pixel_data;
                    data.processed = true;

                    free(tx.pix_info.data.imageData);

                    imgData.imageArrays[tx.pix_info.backingDataIndex] = data;
                    tx.pix_info.data = imgData.imageArrays[tx.pix_info.backingDataIndex];

                    fclose(read_fp);
                } else {
                    printf("Did not find texture in cache - need to construct\n");

                    auto processOutput = process_tex_func(tx);

                    if(!processOutput.shouldCache)
                        continue;

                    free(tx.pix_info.data.imageData);

                    imgData.imageArrays[tx.pix_info.backingDataIndex] = processOutput.newTex;
                    tx.pix_info.data = imgData.imageArrays[tx.pix_info.backingDataIndex];

                    pixel_data = tx.pix_info.data.imageData;
                    pixel_data_size = tx.pix_info.data.imageSize;

                    // Write this data to the cache
                    FILE *write_fp = fopen(path_to_cached_tex.c_str(), "wb");

                    fwrite(&tx.pix_info.data, sizeof(BackingImageData), 1, write_fp);
                    fwrite(pixel_data, pixel_data_size, 1, write_fp );

                    fclose(write_fp);
                }
            } else {
                bool preProcess = tx.pix_info.data.processed;
                void *oldImageData = tx.pix_info.data.imageData;

                auto processOutput = process_tex_func(tx);

                if(!processOutput.shouldCache)
                        continue;

                if(preProcess != processOutput.newTex.processed) {
                    free(oldImageData);
                    imgData.imageArrays[tx.pix_info.backingDataIndex] =
                            processOutput.newTex;
                }

                tx.pix_info.data = imgData.imageArrays[tx.pix_info.backingDataIndex];
            }
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
        .assetInfos { 0 },
    };

    auto obj_loader = Optional<OBJLoader>::none();
    auto gltf_loader = Optional<GLTFLoader>::none();
#ifdef MADRONA_USD_SUPPORT
    auto usd_loader = Optional<USDLoader>::none();
#endif

    auto embree_loader = Optional<EmbreeLoader>::none();

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
            printf("Load failed\n");
            break;
        }

        uint32_t post_objects_offset = imported.objects.size();

        imported.assetInfos.push_back(SourceAssetInfo{
            post_objects_offset - pre_objects_offset,
            std::string(path_view)
        });
    }

    printf("number of materials = %d\n", (int)imported.materials.size());

    printf("\n");

    if (!load_success) {
        return Optional<ImportedAssets>::none();
    }

    return imported;
}

}
