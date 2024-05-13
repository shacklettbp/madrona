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

namespace madrona::imp {

using namespace math;

bool loadCache(const char* location, DynArray<render::MeshBVH>& bvhs_out){
    FILE *ptr;
    ptr = fopen(location, "rb");

    if(ptr == nullptr){
        return false;
    }

    uint32_t num_bvhs;
    fread(&num_bvhs,sizeof(num_bvhs),1,ptr);
    bvhs_out.reserve(num_bvhs);

    for(CountT i = 0; i < num_bvhs; i++) {
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

        assert(num_verts < 2000000);
        assert(num_nodes < 2000000);
        assert(num_leaves < 2000000);

        DynArray<render::MeshBVH::Node> nodes{num_nodes};
        fread(nodes.data(), sizeof(render::MeshBVH::Node), num_nodes, ptr);

#if 0
        DynArray<render::MeshBVH::LeafGeometry> leaf_geos{num_leaves};
        fread(leaf_geos.data(), sizeof(render::MeshBVH::LeafGeometry), num_leaves, ptr);
#endif

        DynArray<render::MeshBVH::BVHVertex> vertices{num_verts};
        fread(vertices.data(), sizeof(render::MeshBVH::BVHVertex), num_verts, ptr);

#ifdef MADRONA_COMPRESSED_DEINDEXED_TEX
        DynArray<render::MeshBVH::LeafMaterial> leaf_materials{num_verts/3};
        fread(leaf_materials.data(), sizeof(render::MeshBVH::LeafMaterial), num_verts/3, ptr);
#endif

        render::MeshBVH bvh;
        bvh.numNodes = num_nodes;
        bvh.numLeaves = num_leaves;
        bvh.numVerts = num_verts;
        bvh.magic = render::MeshBVH::magicSignature;

        bvh.nodes = nodes.release(true);
        // bvh.leafGeos = leaf_geos.release(true);
        bvh.leafMats = leaf_materials.release(true);
        bvh.vertices = vertices.release(true);
        bvh.rootAABB = aabb_out;
        bvh.materialIDX = material_idx;
        bvhs_out.push_back(bvh);
    }

    fclose(ptr);
    return true;
}

void writeCache(const char* location, DynArray<render::MeshBVH>& bvhs){
    FILE *ptr;
    ptr = fopen(location, "wb");

    uint32_t num_bvhs = bvhs.size();
    fwrite(&num_bvhs, sizeof(uint32_t), 1, ptr);

    for(CountT i = 0; i < bvhs.size(); i++) {
        fwrite(&bvhs[i].numVerts, sizeof(uint32_t), 1, ptr);
        fwrite(&bvhs[i].numNodes, sizeof(uint32_t), 1, ptr);
        fwrite(&bvhs[i].numLeaves, sizeof(uint32_t), 1, ptr);
        fwrite(&bvhs[i].rootAABB, sizeof(math::AABB), 1, ptr);
        fwrite(&bvhs[i].materialIDX, sizeof(int32_t), 1, ptr);

        fwrite(bvhs[i].nodes, sizeof(render::MeshBVH::Node), bvhs[i].numNodes, ptr);
        // fwrite(bvhs[i].leafGeos, sizeof(render::MeshBVH::LeafGeometry), bvhs[i].numLeaves, ptr);
        fwrite(bvhs[i].vertices, sizeof(render::MeshBVH::BVHVertex), bvhs[i].numVerts, ptr);

#ifdef MADRONA_COMPRESSED_DEINDEXED_TEX
        fwrite(bvhs[i].leafMats, sizeof(render::MeshBVH::LeafMaterial), bvhs[i].numVerts/3, ptr);
#endif
    }

    fclose(ptr);
}

Optional<ImportedAssets> ImportedAssets::importFromDisk(
    Span<const char * const> paths, Span<char> err_buf,
    bool one_object_per_asset,
    bool generate_mesh_bvhs)
{
    (void)generate_mesh_bvhs;

    ImportedAssets imported {
        .geoData = GeometryData {
            .positionArrays { 0 },
            .normalArrays { 0 },
            .tangentAndSignArrays { 0 },
            .uvArrays { 0 },
            .materialIndices { 0 },
            .indexArrays { 0 },
            .faceCountArrays { 0 },
            .meshArrays { 0 },
            .meshBVHArrays { 0 },
        },
        .imgData = ImageData {
            .imageArrays {0},
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

    auto embree_loader = Optional<EmbreeLoader>::none();

    char* bvh_cache_dir = getenv("MADRONA_BVH_CACHE_DIR");
    std::filesystem::path cache_dir = "";

    char* bvh_cache = getenv("MADRONA_REGEN_BVH_CACHE");
    bool regen_cache = false;
    if(bvh_cache){
       regen_cache = atoi(bvh_cache);
    }

    if(bvh_cache_dir){
        cache_dir = bvh_cache_dir;
    }

    printf("Asset load progress: \n");

    bool load_success = false;
    for (const char *path : paths) {
        uint32_t pre_objects_offset = imported.objects.size();
        printf(".");
        fflush(stdout);

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

#if 1
        if (generate_mesh_bvhs) {
            std::filesystem::path loaded_path = path;

            bool should_construct = true;

            // Create a mesh BVH for all the meshes in recently loaded objects.
            DynArray<render::MeshBVH> asset_bvhs { 0 };

            if(bvh_cache_dir && !regen_cache) {
                should_construct = !loadCache((cache_dir /
                        loaded_path.filename()).c_str(), asset_bvhs);
                if(should_construct){
                    printf("Missing %s. Reconstructing.\n",loaded_path.filename().c_str());
                }
            }

            if(should_construct) {
                uint32_t post_objects_offset = imported.objects.size();
                for (uint32_t object_idx = pre_objects_offset;
                     object_idx < post_objects_offset; ++object_idx) {
                    SourceObject &obj = imported.objects[object_idx];

                    obj.bvhIndex = (uint32_t) asset_bvhs.size();

                    Optional<render::MeshBVH> bvh = embree_loader->load(obj,imported.materials);
                    assert(bvh.has_value());

                    asset_bvhs.push_back(*bvh);
                }
            }

            if(bvh_cache_dir && (regen_cache || should_construct)){
                printf("Caching File %s\n",loaded_path.filename().c_str());
                writeCache((cache_dir / loaded_path.filename()).c_str(), asset_bvhs);
            }

            imported.geoData.meshBVHArrays.push_back(std::move(asset_bvhs));
        }
#endif
    }

    printf("number of materials = %d\n", (int)imported.materials.size());

    printf("\n");

    if (!load_success) {
        return Optional<ImportedAssets>::none();
    }

    return imported;
}

Optional<ImportedAssets::GPUGeometryData> ImportedAssets::makeGPUData(
    const ImportedAssets &assets)
{
#ifdef MADRONA_CUDA_SUPPORT
    using render::MeshBVH;

    uint32_t num_bvhs = 0;
    uint32_t num_nodes = 0;
    uint32_t num_leaf_geos = 0;
    uint32_t num_leaf_mats = 0;
    uint32_t num_vertices = 0;

    for (CountT asset_idx = 0; 
            asset_idx < assets.geoData.meshBVHArrays.size(); 
            ++asset_idx) {
        const DynArray<MeshBVH> &asset_bvhs = 
            assets.geoData.meshBVHArrays[asset_idx];

        num_bvhs += (uint32_t)asset_bvhs.size();

        for (CountT bvh_idx = 0; bvh_idx < asset_bvhs.size(); ++bvh_idx) {
            const MeshBVH &bvh = asset_bvhs[bvh_idx];

            num_nodes += bvh.numNodes;
            num_leaf_geos += bvh.numLeaves;
#ifdef MADRONA_COMPRESSED_DEINDEXED_TEX
            num_leaf_mats += bvh.numVerts/3;
#endif
            num_vertices += bvh.numVerts;
        }
    }

    uint32_t num_bvh_bytes = num_bvhs *
        sizeof(MeshBVH);
    uint32_t num_nodes_bytes = num_nodes *
        sizeof(MeshBVH::Node);
    uint32_t num_leaf_geos_bytes = num_leaf_geos *
        sizeof(MeshBVH::LeafGeometry);
    uint32_t num_leaf_mat_bytes = num_leaf_mats *
        sizeof(MeshBVH::LeafMaterial);
    uint32_t num_vertices_bytes = num_vertices *
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

    uint32_t bvh_offset = 0;
    uint32_t node_offset = 0;
    uint32_t leaf_offset = 0;
    uint32_t vert_offset = 0;

    for (CountT asset_idx = 0; 
            asset_idx < assets.geoData.meshBVHArrays.size(); 
            ++asset_idx) {
        const DynArray<MeshBVH> &asset_bvhs = 
            assets.geoData.meshBVHArrays[asset_idx];

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
            leaf_offset += bvh.numLeaves;
            vert_offset += bvh.numVerts;
        }
    }

    GPUGeometryData gpu_data = {
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

}
