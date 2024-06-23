#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

#include <madrona/mesh_bvh.hpp>
#include <madrona/render/asset_processor.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/mesh_bvh_builder.hpp>

#include <filesystem>

#include <stb_image.h>

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

using namespace madrona::imp;

namespace madrona::render {

namespace AssetProcessor {

#ifdef MADRONA_CUDA_SUPPORT
static bool loadCache(const char *location,
                      HeapArray<MeshBVH> &bvhs_out)
{
    FILE *ptr;
    ptr = fopen(location, "rb");

    if(ptr == nullptr){
        return false;
    }

    const CountT num_bvhs = bvhs_out.size();
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
        bvhs_out.emplace(i, bvh);
    }

    fclose(ptr);
    return true;
}

static void writeCache(const char *location, HeapArray<MeshBVH> &bvhs)
{
    FILE *ptr;
    ptr = fopen(location, "wb");

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

static HeapArray<MeshBVH> createMeshBVHs(
    Span<const SourceObject> objs)
{
    char *bvh_cache_path = getenv("MADRONA_BVH_CACHE");

    bool regen_cache = false;
    {
       char *regen_env = getenv("MADRONA_REGEN_BVH_CACHE");
       if (regen_env) {
           regen_cache = atoi(regen_env);
       }
    }

    HeapArray<MeshBVH> mesh_bvhs(objs.size());

    if (bvh_cache_path && !regen_cache) {
        bool valid_cache = loadCache(bvh_cache_path, mesh_bvhs);

        if (valid_cache) {
            return mesh_bvhs;
        }
    }

    for (CountT obj_idx = 0; obj_idx < objs.size(); obj_idx++) {
        const SourceObject &obj = objs[obj_idx];

        MeshBVH bvh = MeshBVHBuilder::build(obj.meshes);
        mesh_bvhs[obj_idx] = bvh;
    }

     if (bvh_cache_path) {
         writeCache(bvh_cache_path, mesh_bvhs);
     }

    return mesh_bvhs;
}

MeshBVHData makeBVHData(Span<const imp::SourceObject> src_objs)
{
    HeapArray<MeshBVH> mesh_bvhs = createMeshBVHs(src_objs);

    uint64_t num_bvhs = (uint32_t)mesh_bvhs.size();
    uint64_t num_nodes = 0;
    uint64_t num_leaf_mats = 0;
    uint64_t num_vertices = 0;

    for (CountT bvh_idx = 0; bvh_idx < mesh_bvhs.size(); ++bvh_idx) {
        const MeshBVH &bvh = mesh_bvhs[bvh_idx];

        num_nodes += bvh.numNodes;
#ifdef MADRONA_COMPRESSED_DEINDEXED_TEX
        num_leaf_mats += bvh.numVerts/3;
#endif
        num_vertices += bvh.numVerts;
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

    uint64_t node_offset = 0;
    uint64_t leaf_offset = 0;
    uint64_t vert_offset = 0;

    for (CountT bvh_idx = 0; bvh_idx < mesh_bvhs.size(); ++bvh_idx) {
        const MeshBVH &bvh = mesh_bvhs[bvh_idx];

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

        REQ_CUDA(cudaMemcpy(bvhs + bvh_idx,
            &tmp, sizeof(tmp), cudaMemcpyHostToDevice));
        REQ_CUDA(cudaMemcpy(nodes + node_offset,
            bvh.nodes, bvh.numNodes * sizeof(MeshBVH::Node),
            cudaMemcpyHostToDevice));
#if !defined(MADRONA_COMPRESSED_DEINDEXED) && !defined(MADRONA_COMPRESSED_DEINDEXED_TEX)
        REQ_CUDA(cudaMemcpy(leaf_geos + leaf_offset,
            bvh.leafGeos, bvh.numLeaves * sizeof(MeshBVH::LeafGeometry),
            cudaMemcpyHostToDevice));
#endif
#if defined(MADRONA_COMPRESSED_DEINDEXED_TEX)
        REQ_CUDA(cudaMemcpy(leaf_mats + leaf_offset,
            bvh.leafMats, (bvh.numVerts / 3) *
            sizeof(MeshBVH::LeafMaterial),
            cudaMemcpyHostToDevice));
#endif

        REQ_CUDA(cudaMemcpy(vertices + vert_offset,
                    bvh.vertices, bvh.numVerts * sizeof(MeshBVH::BVHVertex),
                    cudaMemcpyHostToDevice));

        node_offset += bvh.numNodes;
        leaf_offset += bvh.numVerts/3;
        vert_offset += bvh.numVerts;
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
        .numBVHs = (uint64_t)mesh_bvhs.size(),
    };

    return gpu_data;
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

        if (tex.format == imp::SourceTextureFormat::BC7) {
            width = tex.width;
            height = tex.height;

            cudaChannelFormatDesc channel_desc =
                    cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed7>();

            cudaArray_t cuda_array;
            REQ_CUDA(cudaMallocArray(&cuda_array, &channel_desc,
                                     width, height, cudaArrayDefault));

            REQ_CUDA(cudaMemcpy2DToArray(cuda_array, 0, 0, tex.data,
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

            pixels = stbi_load_from_memory((stbi_uc*)tex.data,
                                 tex.numBytes, &width,
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
#endif

math::AABB *makeAABBs(
        Span<const imp::SourceObject> src_objs)
{
    int num_objects = (int)src_objs.size();

    math::AABB *aabbs = (math::AABB *)malloc(sizeof(math::AABB) *
            num_objects);

    for (int obj_idx = 0; obj_idx < num_objects; ++obj_idx) {
        auto &obj = src_objs[obj_idx];

        float min_x = FLT_MAX, min_y = FLT_MAX, min_z = FLT_MAX;
        float max_x = -FLT_MAX, max_y = -FLT_MAX, max_z = -FLT_MAX;

        for (int mesh_idx = 0; mesh_idx < (int)obj.meshes.size(); ++mesh_idx) {
            auto &mesh = obj.meshes[mesh_idx];

            for (int v_i = 0; v_i < (int)mesh.numVertices; ++v_i) {
                auto &v = mesh.positions[v_i];

                min_x = std::min(min_x, v.x);
                min_y = std::min(min_y, v.y);
                min_z = std::min(min_z, v.z);

                max_x = std::max(max_x, v.x);
                max_y = std::max(max_y, v.y);
                max_z = std::max(max_z, v.z);
            }
        }

        aabbs[obj_idx] = math::AABB {
            { min_x, min_y, min_z },
            { max_x, max_y, max_z }
        };
    }

    return aabbs;
}
    
}

}
