#pragma once

#include <madrona/render/cuda_batch_render_assets.hpp>

#include <madrona/importer.hpp>
#include <madrona/mesh_bvh.hpp>

namespace madrona::render {

namespace AssetProcessor {
    Optional<MeshBVHData> makeBVHData(
            const imp::ImportedAssets &assets);

    MaterialData initMaterialData(
        const imp::SourceMaterial *materials,
        uint32_t num_materials,
        const imp::SourceTexture *textures,
        uint32_t num_textures);
};

}
