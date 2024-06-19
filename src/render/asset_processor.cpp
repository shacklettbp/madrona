#include <filesystem>
#include <madrona/mesh_bvh.hpp>
#include <madrona/cuda_utils.hpp>
#include <madrona/render/asset_processor.hpp>

#include <madrona/mesh_bvh_builder.hpp>

#include <stb_image.h>

using namespace madrona::imp;

namespace madrona::render {

namespace AssetProcessor {

static bool loadCache(const char* location, DynArray<MeshBVH>& bvhs_out)
{
    FILE *ptr;
    ptr = fopen(location, "rb");

    if(ptr == nullptr){
        return false;
    }

    uint32_t num_bvhs;
    fread(&num_bvhs,sizeof(num_bvhs),1,ptr);
    bvhs_out.reserve(num_bvhs);

    for (CountT i = 0; i < num_bvhs; i++) {
        uint32_t num_verts;
        uint32_t num_nodes;
        uint32_t num_leaves;
        math::AABB aabb_out;
        int32_t material_idx;
        fread(&num_verts, sizeof(num_verts), 1, ptr);
        fread(&num_nodes, sizeof(num_nodes), 1, ptr);
        fread(&num_leaves, sizeof(num_leaves), 1, ptr);
        fread(&aabb_out, sizeof(aabb_out), 1, ptr);
        fread(&material_idx, sizeof(material_idx), 1, ptr);

        DynArray<MeshBVH::Node> nodes{num_nodes};
        fread(nodes.data(), sizeof(MeshBVH::Node), num_nodes, ptr);

#if 0
        DynArray<MeshBVH::LeafGeometry> leaf_geos{num_leaves};
        fread(leaf_geos.data(), sizeof(MeshBVH::LeafGeometry), num_leaves, ptr);
#endif

        DynArray<MeshBVH::BVHVertex> vertices{num_verts};
        fread(vertices.data(), sizeof(MeshBVH::BVHVertex), num_verts, ptr);



#ifdef MADRONA_COMPRESSED_DEINDEXED_TEX
        DynArray<MeshBVH::LeafMaterial> leaf_materials{num_verts/3};
        fread(leaf_materials.data(), sizeof(MeshBVH::LeafMaterial), num_verts/3, ptr);
#endif

        MeshBVH bvh;
        bvh.numNodes = num_nodes;
        bvh.numLeaves = num_leaves;
        bvh.numVerts = num_verts;

        bvh.nodes = nodes.retrieve_ptr();
        // bvh.leafGeos = leaf_geos.release(true);
        bvh.leafMats = leaf_materials.retrieve_ptr();
        bvh.vertices = vertices.retrieve_ptr();
        bvh.rootAABB = aabb_out;
        bvh.materialIDX = material_idx;
        bvhs_out.push_back(bvh);
    }

    fclose(ptr);
    return true;
}

static void writeCache(const char* location, DynArray<MeshBVH>& bvhs)
{
    FILE *ptr;
    ptr = fopen(location, "wb");

    uint32_t num_bvhs = bvhs.size();
    fwrite(&num_bvhs, sizeof(uint32_t), 1, ptr);

    for (CountT i = 0; i < bvhs.size(); i++) {
        fwrite(&bvhs[i].numVerts, sizeof(uint32_t), 1, ptr);
        fwrite(&bvhs[i].numNodes, sizeof(uint32_t), 1, ptr);
        fwrite(&bvhs[i].numLeaves, sizeof(uint32_t), 1, ptr);
        fwrite(&bvhs[i].rootAABB, sizeof(math::AABB), 1, ptr);
        fwrite(&bvhs[i].materialIDX, sizeof(int32_t), 1, ptr);

        fwrite(bvhs[i].nodes, sizeof(MeshBVH::Node), bvhs[i].numNodes, ptr);
        // fwrite(bvhs[i].leafGeos, sizeof(MeshBVH::LeafGeometry), bvhs[i].numLeaves, ptr);
        fwrite(bvhs[i].vertices, sizeof(MeshBVH::BVHVertex), bvhs[i].numVerts, ptr);

#ifdef MADRONA_COMPRESSED_DEINDEXED_TEX
        fwrite(bvhs[i].leafMats, sizeof(MeshBVH::LeafMaterial), bvhs[i].numVerts/3, ptr);
#endif
    }

    fclose(ptr);
}

static DynArray<DynArray<MeshBVH>> createMeshBVHs(const ImportedAssets &assets)
{
    char* bvh_cache_dir = getenv("MADRONA_BVH_CACHE_DIR");
    std::filesystem::path cache_dir = "";

    char* bvh_cache = getenv("MADRONA_REGEN_BVH_CACHE");
    bool regen_cache = false;
    if (bvh_cache) {
       regen_cache = atoi(bvh_cache);
    }

    if (bvh_cache_dir) {
        cache_dir = bvh_cache_dir;
    }

    DynArray<DynArray<MeshBVH>> mesh_bvh_arrays { 0 };

    uint32_t obj_offset = 0;

    for (int asset_idx = 0; asset_idx < assets.assetInfos.size(); ++asset_idx) {
        const SourceAssetInfo &asset_info = assets.assetInfos[asset_idx];

        std::filesystem::path loaded_path = asset_info.path;

        bool should_construct = true;

        // Create a mesh BVH for all the meshes in recently loaded objects.
        DynArray<MeshBVH> asset_bvhs { 0 };

        if(bvh_cache_dir && !regen_cache) {
            should_construct = !loadCache((cache_dir /
                    loaded_path.filename()).c_str(), asset_bvhs);
        }

        if(should_construct) {
            for (uint32_t obj_idx = 0; obj_idx < asset_info.numObjects; ++obj_idx) {
                const SourceObject &obj =
                    assets.objects[obj_offset + obj_idx];

                MeshBVH bvh = MeshBVHBuilder::build(obj.meshes);

                asset_bvhs.push_back(bvh);
            }
        }

        if(bvh_cache_dir && (regen_cache || should_construct)){
            writeCache((cache_dir / loaded_path.filename()).c_str(), 
                    asset_bvhs);
        }

        mesh_bvh_arrays.push_back(std::move(asset_bvhs));

        obj_offset += asset_info.numObjects;
    }

    return mesh_bvh_arrays;
}

Optional<MeshBVHData> makeBVHData(
    const ImportedAssets &assets)
{
    DynArray<DynArray<MeshBVH>> mesh_bvh_arrays =
        createMeshBVHs(assets);

#ifdef MADRONA_CUDA_SUPPORT
    uint64_t num_bvhs = 0;
    uint64_t num_nodes = 0;
    uint64_t num_leaf_mats = 0;
    uint64_t num_vertices = 0;

    for (CountT asset_idx = 0; 
            asset_idx < mesh_bvh_arrays.size(); 
            ++asset_idx) {
        const DynArray<MeshBVH> &asset_bvhs = 
            mesh_bvh_arrays[asset_idx];

        num_bvhs += (uint32_t)asset_bvhs.size();

        for (CountT bvh_idx = 0; bvh_idx < asset_bvhs.size(); ++bvh_idx) {
            const MeshBVH &bvh = asset_bvhs[bvh_idx];

            num_nodes += bvh.numNodes;
#ifdef MADRONA_COMPRESSED_DEINDEXED_TEX
            num_leaf_mats += bvh.numVerts/3;
#endif
            num_vertices += bvh.numVerts;
        }
    }

    uint64_t num_bvh_bytes = num_bvhs *
        sizeof(MeshBVH);
    uint64_t num_nodes_bytes = num_nodes *
        sizeof(MeshBVH::Node);
    uint64_t num_leaf_mat_bytes = num_leaf_mats *
        sizeof(MeshBVH::LeafMaterial);
    uint64_t num_vertices_bytes = num_vertices *
        sizeof(MeshBVH::BVHVertex);

    // All pointers to GPU memory
    auto *bvhs = (MeshBVH *)cu::allocGPU(num_bvh_bytes);
    auto *nodes = (MeshBVH::Node *)cu::allocGPU(num_nodes_bytes);
#if defined(MADRONA_COMPRESSED_DEINDEXED) || defined(MADRONA_COMPRESSED_DEINDEXED_TEX)
    MeshBVH::LeafGeometry *leaf_geos = nullptr;
#else
    MeshBVH::LeafGeometry *leaf_geos = (MeshBVH::LeafGeometry *)
        cu::allocGPU(num_leaf_geos_bytes);
#endif
#if defined(MADRONA_COMPRESSED_DEINDEXED_TEX)
    MeshBVH::LeafMaterial *leaf_mats =(MeshBVH::LeafMaterial*)cu::allocGPU(num_leaf_mat_bytes);
#else
    MeshBVH::LeafMaterial *leaf_mats = nullptr;
#endif

    MeshBVH::BVHVertex *vertices = (MeshBVH::BVHVertex *)cu::allocGPU(num_vertices_bytes);

    uint64_t bvh_offset = 0;
    uint64_t node_offset = 0;
    uint64_t leaf_offset = 0;
    uint64_t vert_offset = 0;

    for (CountT asset_idx = 0; 
            asset_idx < mesh_bvh_arrays.size(); 
            ++asset_idx) {
        const DynArray<MeshBVH> &asset_bvhs = 
            mesh_bvh_arrays[asset_idx];

        for (CountT bvh_idx = 0; bvh_idx < asset_bvhs.size(); ++bvh_idx) {
            const MeshBVH &bvh = asset_bvhs[bvh_idx];

            // Need to make sure the pointers in the BVH point to GPU memory
            MeshBVH tmp = bvh;
            tmp.nodes = nodes + node_offset;
            tmp.leafGeos = leaf_geos + leaf_offset;
            tmp.vertices = vertices + vert_offset;
#if defined(MADRONA_COMPRESSED_DEINDEXED_TEX)
            tmp.leafMats = leaf_mats + leaf_offset;
#else
            tmp.leafMats = nullptr;
#endif

            REQ_CUDA(cudaMemcpy(bvhs + bvh_offset,
                        &tmp, sizeof(tmp), cudaMemcpyHostToDevice));
            REQ_CUDA(cudaMemcpy(nodes + node_offset,
                        bvh.nodes, bvh.numNodes * sizeof(MeshBVH::Node),
                        cudaMemcpyHostToDevice));
#if !defined(MADRONA_COMPRESSED_DEINDEXED) && !defined(MADRONA_COMPRESSED_DEINDEXED_TEX)
            REQ_CUDA(cudaMemcpy(leaf_geos + leaf_offset,
                        bvh.leafGeos, bvh.numLeaves * 
                            sizeof(MeshBVH::LeafGeometry),
                        cudaMemcpyHostToDevice));
#endif
#if defined(MADRONA_COMPRESSED_DEINDEXED_TEX)
            REQ_CUDA(cudaMemcpy(leaf_mats + leaf_offset,
                        bvh.leafMats, (bvh.numVerts/3) *
                            sizeof(MeshBVH::LeafMaterial),
                        cudaMemcpyHostToDevice));
#endif

            REQ_CUDA(cudaMemcpy(vertices + vert_offset,
                        bvh.vertices, bvh.numVerts * sizeof(MeshBVH::BVHVertex),
                        cudaMemcpyHostToDevice));

            bvh_offset += 1;
            node_offset += bvh.numNodes;
            leaf_offset += bvh.numVerts/3;
            vert_offset += bvh.numVerts;
        }
    }

    MeshBVHData gpu_data = {
        .nodes = nodes,
        .numNodes = node_offset,
        .leafGeos = leaf_geos,
        .leafMaterial = leaf_mats,
        .numLeaves = leaf_offset,
        .vertices = vertices,
        .numVerts = vert_offset,
        .meshBVHs = bvhs,
        .numBVHs = bvh_offset,
    };

    return gpu_data;
#else
    return {};
#endif
}

MaterialData initMaterialData(
    const imp::SourceMaterial *materials,
    uint32_t num_materials,
    const imp::SourceTexture *textures,
    uint32_t num_textures)
{
    MaterialData cpu_mat_data = {
        .textures = (cudaTextureObject_t *)
            malloc(sizeof(cudaTextureObject_t) * num_textures),
        .textureBuffers = (cudaArray_t *)
            malloc(sizeof(cudaArray_t) * num_textures),
        .materials = (Material *)
            malloc(sizeof(Material) * num_materials)
    };

    for (uint32_t i = 0; i < num_textures; ++i) {
        const auto &tex = textures[i];
        int width, height, components;
        void *pixels = nullptr;

        if (tex.config.format == imp::TextureFormat::BC7) {
            width = tex.config.width;
            height = tex.config.height;

            cudaChannelFormatDesc channel_desc =
                    cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed7>();

            cudaArray_t cuda_array;
            REQ_CUDA(cudaMallocArray(&cuda_array, &channel_desc,
                                     width, height, cudaArrayDefault));

            REQ_CUDA(cudaMemcpy2DToArray(cuda_array, 0, 0, tex.imageData,
                                         16 * width / 4,
                                         16 * width / 4,
                                         height / 4,
                                         cudaMemcpyHostToDevice));

            cudaResourceDesc res_desc = {};
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = cuda_array;

            cudaTextureDesc tex_desc = {};
            tex_desc.addressMode[0] = cudaAddressModeWrap;
            tex_desc.addressMode[1] = cudaAddressModeWrap;
            tex_desc.filterMode = cudaFilterModeLinear;
            tex_desc.readMode = cudaReadModeNormalizedFloat;
            tex_desc.normalizedCoords = 1;

            cudaTextureObject_t tex_obj = 0;
            REQ_CUDA(cudaCreateTextureObject(&tex_obj,
                                             &res_desc, &tex_desc, nullptr));

            cpu_mat_data.textures[i] = tex_obj;
            cpu_mat_data.textureBuffers[i] = cuda_array;
        } else {

            pixels = stbi_load_from_memory((stbi_uc*)tex.imageData,
                                 tex.config.imageSize, &width,
                                 &height, &components, STBI_rgb_alpha);
            // For now, only allow this format
            cudaChannelFormatDesc channel_desc =
                cudaCreateChannelDesc<uchar4>();


            cudaArray_t cuda_array;
            REQ_CUDA(cudaMallocArray(&cuda_array, &channel_desc,
                                     width, height, cudaArrayDefault));

            REQ_CUDA(cudaMemcpy2DToArray(cuda_array, 0, 0, pixels,
                                       sizeof(uint32_t) * width,
                                       sizeof(uint32_t) * width,
                                       height,
                                       cudaMemcpyHostToDevice));

            cudaResourceDesc res_desc = {};
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = cuda_array;

            cudaTextureDesc tex_desc = {};
            tex_desc.addressMode[0] = cudaAddressModeWrap;
            tex_desc.addressMode[1] = cudaAddressModeWrap;
            tex_desc.filterMode = cudaFilterModeLinear;
            tex_desc.readMode = cudaReadModeNormalizedFloat;
            tex_desc.normalizedCoords = 1;

            cudaTextureObject_t tex_obj = 0;
            REQ_CUDA(cudaCreateTextureObject(&tex_obj,
                        &res_desc, &tex_desc, nullptr));

            cpu_mat_data.textures[i] = tex_obj;
            cpu_mat_data.textureBuffers[i] = cuda_array;
        }
    }

    for (uint32_t i = 0; i < num_materials; ++i) {
        Material mat = {
            .color = materials[i].color,
            .textureIdx = materials[i].textureIdx,
            .roughness = materials[i].roughness,
            .metalness = materials[i].metalness,
        };

        cpu_mat_data.materials[i] = mat;
    }

    cudaTextureObject_t *gpu_tex_buffer;
    REQ_CUDA(cudaMalloc(&gpu_tex_buffer, 
                sizeof(cudaTextureObject_t) * num_textures));
    REQ_CUDA(cudaMemcpy(gpu_tex_buffer, cpu_mat_data.textures, 
                sizeof(cudaTextureObject_t) * num_textures,
                cudaMemcpyHostToDevice));

    Material *mat_buffer;
    REQ_CUDA(cudaMalloc(&mat_buffer, 
                sizeof(Material) * num_materials));
    REQ_CUDA(cudaMemcpy(mat_buffer, cpu_mat_data.materials, 
                sizeof(Material) * num_materials,
                cudaMemcpyHostToDevice));

    free(cpu_mat_data.textures);
    free(cpu_mat_data.materials);

    auto gpu_mat_data = cpu_mat_data;
    gpu_mat_data.textures = gpu_tex_buffer;
    gpu_mat_data.materials = mat_buffer;

    return gpu_mat_data;
}
    
}

}
