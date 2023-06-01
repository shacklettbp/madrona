#include "scene.hpp"
#include "asset_utils.hpp"

namespace madrona::render::metal {

static inline MTL::Heap * makeAssetHeap(MTL::Device *dev,
                                        int64_t heap_size)
{
    MTL::HeapDescriptor *heap_desc = MTL::HeapDescriptor::alloc()->init();
    heap_desc->setType(MTL::HeapTypePlacement);
    heap_desc->setStorageMode(MTL::StorageModePrivate);
    heap_desc->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    heap_desc->setSize(heap_size);

    MTL::Heap *heap = dev->newHeap(heap_desc);
    heap_desc->release();

    return heap;
}

AssetManager::AssetManager(MTL::Device *dev)
    : transferQueue(dev->newCommandQueue())
{}

Optional<AssetMetadata> AssetManager::prepareSourceAssets(
    Span<const imp::SourceObject> src_objs)
{
    using namespace imp;

    int64_t num_total_vertices = 0;
    int64_t num_total_indices = 0;
    int64_t num_total_meshes = 0;

    for (const SourceObject &obj : src_objs) {
        num_total_meshes += obj.meshes.size();

        for (const SourceMesh &mesh : obj.meshes) {
            if (mesh.faceCounts != nullptr) {
                FATAL("Render mesh isn't triangular");
            }

            num_total_vertices += mesh.numVertices;
            num_total_indices += mesh.numFaces * 3;
        }
    }

    int64_t num_total_objs = src_objs.size();

    int64_t offsets[3];
    int64_t num_asset_bytes = utils::computeBufferOffsets({
            (int64_t)sizeof(ObjectData) * num_total_objs,
            (int64_t)sizeof(MeshData) * num_total_meshes,
            (int64_t)sizeof(PackedVertex) * num_total_vertices,
            (int64_t)sizeof(uint32_t) * num_total_indices,
        }, offsets, consts::mtlBufferAlignment);

    return AssetMetadata {
        offsets[0],
        offsets[1],
        offsets[2],
        num_asset_bytes,
    };
}

void AssetManager::packSourceAssets(
    void *dst_buf,
    const AssetMetadata &metadata,
    Span<const imp::SourceObject> src_objs)
{
    using namespace math;
    using namespace imp;

    char *base_ptr = (char *)dst_buf;
    auto *obj_ptr = (ObjectData *)base_ptr;
    auto *mesh_ptr = (MeshData *)(base_ptr + metadata.meshesOffset);
    auto *vertex_ptr = (PackedVertex *)(base_ptr + metadata.verticesOffset);
    auto *indices_ptr = (uint32_t *)(base_ptr + metadata.indicesOffset);

    int32_t mesh_offset = 0;
    int32_t vertex_offset = 0;
    int32_t index_offset = 0;
    for (const SourceObject &obj : src_objs) {
        *obj_ptr++ = ObjectData {
            .meshOffset = mesh_offset,
            .numMeshes = (int32_t)obj.meshes.size(),
        };

        for (const SourceMesh &mesh : obj.meshes) {
            int32_t num_mesh_verts = (int32_t)mesh.numVertices;
            int32_t num_mesh_indices = (int32_t)mesh.numFaces * 3;

            mesh_ptr[mesh_offset++] = MeshData {
                .vertexOffset = vertex_offset,
                .numVertices = (int32_t)num_mesh_verts,
                .indexOffset = index_offset,
                .numIndices = (int32_t)num_mesh_indices,
            };

            // Compute new normals
            auto new_normals = Optional<HeapArray<Vector3>>::none();
            if (!mesh.normals) {
                new_normals.emplace(num_mesh_verts);

                for (int64_t vert_idx = 0; vert_idx < num_mesh_verts;
                     vert_idx++) {
                    (*new_normals)[vert_idx] = Vector3::zero();
                }

                for (CountT face_idx = 0; face_idx < (CountT)mesh.numFaces;
                     face_idx++) {
                    CountT base_idx = face_idx * 3;
                    uint32_t i0 = mesh.indices[base_idx];
                    uint32_t i1 = mesh.indices[base_idx + 1];
                    uint32_t i2 = mesh.indices[base_idx + 2];

                    Vector3 v0 = mesh.positions[i0];
                    Vector3 v1 = mesh.positions[i1];
                    Vector3 v2 = mesh.positions[i2];

                    Vector3 e0 = v1 - v0;
                    Vector3 e1 = v2 - v0;

                    Vector3 face_normal = cross(e0, e1);
                    float face_len = face_normal.length();
                    assert(face_len != 0);
                    face_normal /= face_len;

                    (*new_normals)[i0] += face_normal;
                    (*new_normals)[i1] += face_normal;
                    (*new_normals)[i2] += face_normal;
                }

                for (int64_t vert_idx = 0; vert_idx < num_mesh_verts;
                     vert_idx++) {
                    (*new_normals)[vert_idx] =
                        normalize((*new_normals)[vert_idx]);
                }
            }

            for (int32_t i = 0; i < num_mesh_verts; i++) {
                Vector3 pos = mesh.positions[i];
                Vector3 normal = mesh.normals ?
                    mesh.normals[i] : (*new_normals)[i];
                Vector4 tangent_sign;
                // FIXME: use mikktspace at import time
                if (mesh.tangentAndSigns != nullptr) {
                    tangent_sign = mesh.tangentAndSigns[i];
                } else {
                    Vector3 a, b;
                    normal.frame(&a, &b);
                    tangent_sign = {
                        a.x,
                        a.y,
                        a.z,
                        1.f,
                    };
                }
                Vector2 uv = mesh.uvs ? mesh.uvs[i] : Vector2 { 0, 0 };

                Vector3 encoded_normal_tangent =
                    encodeNormalTangent(normal, tangent_sign);

                vertex_ptr[vertex_offset++] = PackedVertex {
                    Vector4 {
                        pos.x,
                        pos.y,
                        pos.z,
                        encoded_normal_tangent.x,
                    },
                    Vector4 {
                        encoded_normal_tangent.y,
                        encoded_normal_tangent.z,
                        uv.x,
                        uv.y,
                    },
                };
            }

            memcpy(indices_ptr + index_offset,
                   mesh.indices, sizeof(uint32_t) * num_mesh_indices);

            index_offset += num_mesh_indices;
        }
    }
}

StagedAssets AssetManager::stageSourceAssets(
    MTL::Device *dev,
    const AssetMetadata &metadata,
    Span<const imp::SourceObject> src_objs)
{
    static_assert(sizeof(AssetsArgBuffer) < consts::mtlBufferAlignment);
    int64_t total_bytes = consts::mtlBufferAlignment;

    total_bytes += metadata.numAssetBytes;

    MTL::Buffer *staging = dev->newBuffer(
        total_bytes, MTL::ResourceStorageModeShared |
        MTL::ResourceHazardTrackingModeUntracked |
        MTL::ResourceCPUCacheModeWriteCombined);

    packSourceAssets((char *)staging->contents() + consts::mtlBufferAlignment,
                     metadata, src_objs);

    return StagedAssets {
        .stagingBuffer = staging,
        .numTotalBytes = total_bytes,
    };
}

Assets AssetManager::load(MTL::Device *dev,
                          const AssetMetadata &metadata,
                          StagedAssets &staged)
{
    MTL::Heap *asset_heap = makeAssetHeap(dev, staged.numTotalBytes);
    MTL::Buffer *asset_buffer = asset_heap->newBuffer(
        staged.numTotalBytes, MTL::ResourceStorageModePrivate | 
            MTL::ResourceHazardTrackingModeUntracked, 0);

    uint64_t buffer_base_addr =
        asset_buffer->gpuAddress() + consts::mtlBufferAlignment;

    AssetsArgBuffer *staged_argbuffer =
        (AssetsArgBuffer *)(staged.stagingBuffer->contents());
    *staged_argbuffer = AssetsArgBuffer {
        .vertices = buffer_base_addr + metadata.verticesOffset,
        .indices = buffer_base_addr + metadata.indicesOffset,
        .meshes = buffer_base_addr + metadata.meshesOffset,
        .objects = buffer_base_addr,
    };

    MTL::CommandBuffer *copy_cmd = 
        transferQueue->commandBufferWithUnretainedReferences();
    MTL::BlitCommandEncoder *copy_enc = copy_cmd->blitCommandEncoder();

    copy_enc->copyFromBuffer(staged.stagingBuffer, 0, asset_buffer, 0,
                             staged.numTotalBytes);

    copy_enc->endEncoding();
    copy_cmd->commit();
    copy_cmd->waitUntilCompleted();

    staged.stagingBuffer->release();

    return Assets {
        .heap = asset_heap,
        .buffer = asset_buffer,
    };
}

void AssetManager::free(const Assets &assets)
{
    assets.heap->release();
}

}
