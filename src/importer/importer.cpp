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

#include "embree.hpp"

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

namespace madrona::imp {

using namespace math;

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
            .indexArrays { 0 },
            .faceCountArrays { 0 },
            .meshArrays { 0 },
            .meshBVHArrays { 0 },
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
            break;
        }

#if 1
        if (generate_mesh_bvhs) {
            uint32_t post_objects_offset = imported.objects.size();

            // Create a mesh BVH for all the meshes in recently loaded objects.
            DynArray<render::MeshBVH> asset_bvhs { 0 };

            for (uint32_t object_idx = pre_objects_offset;
                    object_idx < post_objects_offset; ++object_idx) {
                SourceObject &obj = imported.objects[object_idx];

                obj.bvhIndex = (uint32_t)asset_bvhs.size();

                Optional<render::MeshBVH> bvh = embree_loader->load(obj);
                assert(bvh.has_value());

                asset_bvhs.push_back(*bvh);
            }

            imported.geoData.meshBVHArrays.push_back(std::move(asset_bvhs));
        }
#endif
    }

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
            num_vertices += bvh.numVerts;
        }
    }

    uint32_t num_bvh_bytes = num_bvhs *
        sizeof(MeshBVH);
    uint32_t num_nodes_bytes = num_nodes *
        sizeof(MeshBVH::Node);
    uint32_t num_leaf_geos_bytes = num_leaf_geos *
        sizeof(MeshBVH::LeafGeometry);
    uint32_t num_vertices_bytes = num_vertices *
        sizeof(math::Vector3);

    // All pointers to GPU memory
    auto *bvhs = (MeshBVH *)cu::allocGPU(num_bvh_bytes);
    auto *nodes = (MeshBVH::Node *)cu::allocGPU(num_nodes_bytes);
    auto *leaf_geos = (MeshBVH::LeafGeometry *)
        cu::allocGPU(num_leaf_geos_bytes);
    math::Vector3 *vertices = (math::Vector3 *)cu::allocGPU(num_vertices_bytes);

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

            REQ_CUDA(cudaMemcpy(bvhs + bvh_offset,
                        &tmp, sizeof(tmp), cudaMemcpyHostToDevice));
            REQ_CUDA(cudaMemcpy(nodes + node_offset,
                        bvh.nodes, bvh.numNodes * sizeof(MeshBVH::Node),
                        cudaMemcpyHostToDevice));
            REQ_CUDA(cudaMemcpy(leaf_geos + leaf_offset,
                        bvh.leafGeos, bvh.numLeaves * 
                            sizeof(MeshBVH::LeafGeometry),
                        cudaMemcpyHostToDevice));
            REQ_CUDA(cudaMemcpy(vertices + vert_offset,
                        bvh.vertices, bvh.numVerts * sizeof(math::Vector3),
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
        .numLeaves = leaf_offset,
        .vertices = vertices,
        .numVerts = vert_offset,
        .meshBVHs = bvhs,
        .numBVHs = bvh_offset,
    };

    return gpu_data;
#else
    return Optional<ImportedAssets::GPUGeometryData>::none();
#endif
}

}
