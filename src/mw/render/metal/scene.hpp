#pragma once

#include "shader.hpp"
#include "../interop.hpp"

#include <madrona/heap_array.hpp>
#include <madrona/importer.hpp>
#include <madrona/render/mw.hpp>

#include <Metal/Metal.hpp>

namespace madrona::render::metal {

namespace consts {
// https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/2928169-setbuffers
constexpr inline int64_t mtlBufferAlignment = 256;
}

struct AssetMetadata {
    int64_t meshesOffset;
    int64_t verticesOffset;
    int64_t indicesOffset;
    int64_t numAssetBytes;
};

struct StagedAssets {
    MTL::Buffer *stagingBuffer;
    int64_t numTotalBytes;
};

struct Assets {
    MTL::Heap *heap;
    MTL::Buffer *buffer;
};

struct AssetManager {
    MTL::CommandQueue *transferQueue;

    AssetManager(MTL::Device *dev);

    Optional<AssetMetadata> prepareSourceAssets(
        Span<const imp::SourceObject> src_objs);

    void packSourceAssets(
        void *dst_buf,
        const AssetMetadata &metadata,
        Span<const imp::SourceObject> src_objs);

    StagedAssets stageSourceAssets(
        MTL::Device *dev,
        const AssetMetadata &metadata,
        Span<const imp::SourceObject> src_objs);

    Assets load(MTL::Device *dev,
                const AssetMetadata &metadata,
                StagedAssets &staged);

    void free(const Assets &assets);
};

}
