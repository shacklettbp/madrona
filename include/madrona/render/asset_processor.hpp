#pragma once

#include <madrona/render/cuda_batch_render_assets.hpp>

#include <madrona/importer.hpp>
#include <madrona/mesh_bvh.hpp>

namespace madrona::render {

namespace AssetProcessor {
    Optional<MeshBVHData> makeBVHData(
        Span<const imp::SourceObject> src_objs);

    MaterialData initMaterialData(
        const imp::SourceMaterial *materials,
        uint32_t num_materials,
        const imp::SourceTexture *textures,
        uint32_t num_textures);
};

}
