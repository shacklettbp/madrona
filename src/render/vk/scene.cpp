#include "scene.hpp"
#include "core.hpp"
#include "shader.hpp"
#include "utils.hpp"

using namespace std;

namespace madrona {

using namespace math;

namespace render {
namespace vk {

BLASData::BLASData(const DeviceState &d, vector<BLAS> &&as,
                   LocalBuffer &&buf)
    : dev(&d),
      accelStructs(move(as)),
      storage(move(buf))
{}

BLASData::BLASData(BLASData &&o)
    : dev(o.dev),
      accelStructs(move(o.accelStructs)),
      storage(move(o.storage))
{}

static void freeBLASes(const DeviceState &dev, const vector<BLAS> &blases)
{
    for (const auto &blas : blases) {
        dev.dt.destroyAccelerationStructureKHR(dev.hdl, blas.hdl,
                                                nullptr);
    }
}

BLASData &BLASData::operator=(BLASData &&o)
{
    freeBLASes(*dev, accelStructs);

    dev = o.dev;
    accelStructs = move(o.accelStructs);
    storage = move(o.storage);

    return *this;
}

BLASData::~BLASData()
{
    freeBLASes(*dev, accelStructs);
}

struct BLASBuildResults {
    BLASData blases;
    optional<LocalBuffer> scratch;
    optional<HostBuffer> staging;
    VkDeviceSize totalBLASBytes;
    bool blasesRebuilt;
};

static optional<BLASBuildResults> makeBLASes(
    const DeviceState &dev,
    MemoryAllocator &alloc, 
    const AssetMetadata &asset_metadata,
    VkDeviceAddress geo_base,
    VkCommandBuffer build_cmd)
{
    const auto &meshes = asset_metadata.meshes;
    const auto &objects = asset_metadata.objects;
    const auto &object_offsets = asset_metadata.objectOffsets;

    vector<VkAccelerationStructureGeometryKHR> geo_infos;
    vector<uint32_t> triangle_counts;
    vector<VkAccelerationStructureBuildRangeInfoKHR> range_infos;

    geo_infos.reserve(meshes.size());
    triangle_counts.reserve(meshes.size());
    range_infos.reserve(meshes.size());

    vector<VkAccelerationStructureBuildGeometryInfoKHR> build_infos;
    vector<tuple<VkDeviceSize, VkDeviceSize, VkDeviceSize>> memory_locs;

    build_infos.reserve(objects.size());
    memory_locs.reserve(objects.size());

    VkDeviceSize total_scratch_bytes = 0;
    VkDeviceSize total_accel_bytes = 0;

    for (int64_t obj_idx = 0; obj_idx < objects.size(); obj_idx++) {
        const Object &object = objects[obj_idx];
        uint32_t obj_offset = object_offsets[obj_idx];
        VkDeviceAddress obj_dev_addr = geo_base + obj_offset;

        for (int64_t mesh_idx = 0; mesh_idx < (int64_t)object.numMeshes;
             mesh_idx++) {
            const Mesh &mesh = meshes[object.meshOffset + mesh_idx];

            VkAccelerationStructureGeometryKHR geo_info;
            geo_info.sType =
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            geo_info.pNext = nullptr;
            geo_info.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            geo_info.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
            auto &tri_info = geo_info.geometry.triangles;
            tri_info.sType =
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            tri_info.pNext = nullptr;
            tri_info.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            tri_info.vertexData.deviceAddress = obj_dev_addr;
            tri_info.vertexStride = sizeof(shader::PackedVertex);
            tri_info.maxVertex = mesh.vertexOffset + mesh.numVertices;
            tri_info.indexType = VK_INDEX_TYPE_UINT32;
            tri_info.indexData.deviceAddress = obj_dev_addr;
            tri_info.transformData.deviceAddress = 0;

            uint32_t num_triangles = mesh.numIndices / 3;

            geo_infos.push_back(geo_info);
            triangle_counts.push_back(num_triangles);

            VkAccelerationStructureBuildRangeInfoKHR range_info;
            range_info.primitiveCount = num_triangles;
            range_info.primitiveOffset = mesh.indexOffset * sizeof(uint32_t);
            range_info.firstVertex = mesh.vertexOffset;
            range_info.transformOffset = 0;
            range_infos.push_back(range_info);
        }

        VkAccelerationStructureBuildGeometryInfoKHR build_info;
        build_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        build_info.pNext = nullptr;
        build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build_info.flags =
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
            VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
        build_info.mode =
            VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build_info.srcAccelerationStructure = VK_NULL_HANDLE;
        build_info.dstAccelerationStructure = VK_NULL_HANDLE;
        build_info.geometryCount = object.numMeshes;
        build_info.pGeometries = &geo_infos[object.meshOffset];
        build_info.ppGeometries = nullptr;
        // Set device address to 0 before space calculation 
        build_info.scratchData.deviceAddress = 0;
        build_infos.push_back(build_info);

        VkAccelerationStructureBuildSizesInfoKHR size_info;
        size_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        size_info.pNext = nullptr;

        dev.dt.getAccelerationStructureBuildSizesKHR(
            dev.hdl, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &build_infos.back(),
            &triangle_counts[object.meshOffset],
            &size_info);

         // Must be aligned to 256 as per spec
        total_accel_bytes = alignOffset(total_accel_bytes, 256);

        memory_locs.emplace_back(total_scratch_bytes, total_accel_bytes,
                                 size_info.accelerationStructureSize);

        total_scratch_bytes += size_info.buildScratchSize;
        total_accel_bytes += size_info.accelerationStructureSize;
    }

    optional<LocalBuffer> scratch_mem_opt =
        alloc.makeLocalBuffer(total_scratch_bytes, true);

    optional<LocalBuffer> accel_mem_opt =
        alloc.makeLocalBuffer(total_accel_bytes, true);

    if (!scratch_mem_opt.has_value() || !accel_mem_opt.has_value()) {
        return {};
    }

    LocalBuffer &scratch_mem = scratch_mem_opt.value();
    LocalBuffer &accel_mem = accel_mem_opt.value();

    VkBufferDeviceAddressInfoKHR scratch_addr_info;
    scratch_addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    scratch_addr_info.pNext = nullptr;
    scratch_addr_info.buffer = scratch_mem.buffer;
    VkDeviceAddress scratch_base_addr =
        dev.dt.getBufferDeviceAddress(dev.hdl, &scratch_addr_info);

    vector<BLAS> accel_structs;
    vector<VkAccelerationStructureBuildRangeInfoKHR *> range_info_ptrs;
    accel_structs.reserve(objects.size());
    range_info_ptrs.reserve(objects.size());

    for (int obj_idx = 0; obj_idx < (int)objects.size(); obj_idx++) {
        VkAccelerationStructureCreateInfoKHR create_info;
        create_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        create_info.pNext = nullptr;
        create_info.createFlags = 0;
        create_info.buffer = accel_mem.buffer;
        create_info.offset = get<1>(memory_locs[obj_idx]);
        create_info.size = get<2>(memory_locs[obj_idx]);
        create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        create_info.deviceAddress = 0;

        VkAccelerationStructureKHR blas;
        REQ_VK(dev.dt.createAccelerationStructureKHR(dev.hdl, &create_info,
                                                     nullptr, &blas));

        auto &build_info = build_infos[obj_idx];
        build_info.dstAccelerationStructure = blas;
        build_info.scratchData.deviceAddress =
            scratch_base_addr + get<0>(memory_locs[obj_idx]);

        VkAccelerationStructureDeviceAddressInfoKHR addr_info;
        addr_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addr_info.pNext = nullptr;
        addr_info.accelerationStructure = blas;

        VkDeviceAddress dev_addr = 
            dev.dt.getAccelerationStructureDeviceAddressKHR(dev.hdl, &addr_info);

        accel_structs.push_back({
            blas,
            dev_addr,
        });

        range_info_ptrs.push_back(&range_infos[objects[obj_idx].meshOffset]);
    }

    dev.dt.cmdBuildAccelerationStructuresKHR(build_cmd,
        build_infos.size(), build_infos.data(), range_info_ptrs.data());

    return BLASBuildResults {
        BLASData(dev, move(accel_structs), move(accel_mem)),
        move(scratch_mem),
        {},
        total_accel_bytes,
        true,
    };
}

AssetManager::AssetManager(const DeviceState &dev,
                           MemoryAllocator &mem,
                           int cuda_gpu_id,
                           int64_t max_objects)
    : addrBufferStaging(mem.makeStagingBuffer(
            max_objects * (sizeof(uint64_t) + sizeof(shader::ObjectData)))),
      addrBuffer(mem.makeDedicatedBuffer(
            max_objects * (sizeof(uint64_t) + sizeof(shader::ObjectData)))),
      addrBufferCUDA(dev, cuda_gpu_id, addrBuffer.mem,
            max_objects * (sizeof(uint64_t) + sizeof(shader::ObjectData))),
      freeObjectOffset(0),
      maxObjects(max_objects)
{}

static Vector3 encodeNormalTangent(const Vector3 &normal,
                                   const Vector4 &tangent_plussign)
{
    auto packHalf2x16 = [](const Vector2 &v) {
        __fp16 x_half = v.x;
        __fp16 y_half = v.y;

        return uint32_t(std::bit_cast<uint16_t>(x_half)) << 16 |
            uint32_t(std::bit_cast<uint16_t>(y_half));
    };

    auto packSnorm2x16 = [](const Vector2 &v) {
        uint16_t x = roundf(min(max(v.x, -1.f), 1.f) * 32767.f);
        uint16_t y = roundf(min(max(v.y, -1.f), 1.f) * 32767.f);

        return uint32_t(x) << 16 | uint32_t(y);
    };

    // https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
    auto octWrap = [](const Vector2 &v) {
        return Vector2 {
            (1.f - fabsf(v.y)) * (v.x >= 0.f ? 1.f : -1.f),
            (1.f - fabsf(v.x)) * (v.y >= 0.f? 1.f : -1.f),
        };
    };
 
    auto octEncode = [&octWrap](Vector3 n) {
        n /= (fabsf(n.x) + fabsf(n.y) + fabsf(n.z));

        Vector2 nxy {n.x, n.y};

        nxy = n.z >= 0.0f ? nxy : octWrap(nxy);
        nxy = nxy * 0.5f + 0.5f;
        return nxy;
    };

    Vector3 tangent = {
        tangent_plussign.x,
        tangent_plussign.y,
        tangent_plussign.z,
    };
    float bitangent_sign = tangent_plussign.w;

    uint32_t nxy = packHalf2x16(Vector2 {normal.x, normal.y});
    uint32_t nzsign = packHalf2x16(Vector2 {normal.z, bitangent_sign});

    Vector2 octtan = octEncode(tangent);
    uint32_t octtan_snorm = packSnorm2x16(octtan);

    return Vector3 {
        std::bit_cast<float>(nxy),
        std::bit_cast<float>(nzsign),
        std::bit_cast<float>(octtan_snorm),
    };
}

static shader::PackedVertex packVertex(const SourceVertex &v)
{
    Vector3 encoded_normal_tangent =
        encodeNormalTangent(v.normal, v.tangentAndSign);

    return shader::PackedVertex {
        Vector4 {
            v.position.x,
            v.position.y,
            v.position.z,
            encoded_normal_tangent.x,
        },
        Vector4 {
            encoded_normal_tangent.y,
            encoded_normal_tangent.z,
            v.uv.x,
            v.uv.y,
        },
    };
}

Optional<AssetMetadata> AssetManager::prepareMetadata(
        Span<const SourceObject> objects)
{
    uint64_t total_bytes = 0;
    int64_t total_num_meshes = 0;
    for (const SourceObject &obj : objects) {
        total_bytes += sizeof(shader::MeshData) * obj.meshes.size();
        total_bytes = alignOffset(total_bytes, sizeof(shader::PackedVertex));

        total_num_meshes += obj.meshes.size();
        for (const SourceMesh &mesh : obj.meshes) {
            total_bytes += mesh.vertices.size() * sizeof(shader::PackedVertex);
            total_bytes += mesh.indices.size() * sizeof(uint32_t);
        }

        total_bytes = alignOffset(total_bytes, sizeof(shader::PackedVertex));
    }

    if (total_bytes >= (uint64_t(1) << uint64_t(31))) {
        return Optional<AssetMetadata>::none();
    }

    return AssetMetadata {
        HeapArray<Mesh>(total_num_meshes),
        HeapArray<Object>(objects.size()),
        HeapArray<uint32_t>(objects.size()),
        uint32_t(total_bytes),
    };
}

void AssetManager::packAssets(void *dst_buffer,
                              AssetMetadata &metadata,
                              Span<const SourceObject> src_objects)
{
    char *asset_pack_base = (char *)dst_buffer;
    uint32_t pack_offset = 0;

    int64_t obj_mesh_offset = 0;
    for (int64_t obj_idx = 0; obj_idx < src_objects.size(); obj_idx++) {
        const SourceObject &src_obj = src_objects[obj_idx];
        int64_t num_meshes = src_obj.meshes.size();

        Object &dst_obj = metadata.objects[obj_idx];
        dst_obj.meshOffset = uint32_t(obj_mesh_offset);
        dst_obj.numMeshes = uint32_t(num_meshes);

        // Base offset for this object
        metadata.objectOffsets[obj_idx] = pack_offset;

        // Base pointer for packing GPU data for this object
        char *obj_pack_base = asset_pack_base + pack_offset;

        // shader::MeshData array is stored first
        auto *obj_meshdata = (shader::MeshData *)obj_pack_base;
        
        // Vertex data will get stored after all the shader::MeshData structs
        // so precalculate how much we need to skip ahead by
        int64_t num_meshdata_bytes = utils::roundUpPow2(
            num_meshes * sizeof(shader::MeshData), 
            sizeof(shader::PackedVertex));

        int64_t cur_vertex_offset =
            num_meshdata_bytes / sizeof(shader::PackedVertex);

        shader::PackedVertex *cur_vertex_ptr =
            (shader::PackedVertex *)(obj_pack_base + num_meshdata_bytes);

        for (int64_t mesh_idx = 0; mesh_idx < num_meshes; mesh_idx++) {
            const SourceMesh &src_mesh = src_obj.meshes[mesh_idx];
            Mesh &dst_mesh = metadata.meshes[obj_mesh_offset + mesh_idx];

            int64_t num_vertices = src_mesh.vertices.size();

            dst_mesh.vertexOffset = uint32_t(cur_vertex_offset); // FIXME
            dst_mesh.numVertices = uint32_t(num_vertices);
            obj_meshdata[mesh_idx].vertexOffset = uint32_t(cur_vertex_offset);

            for (int64_t vert_idx = 0; vert_idx < num_vertices; vert_idx++) {
                cur_vertex_ptr[vert_idx] =
                    packVertex(src_mesh.vertices[vert_idx]);
            }

            cur_vertex_ptr += num_vertices;
            cur_vertex_offset += num_vertices;
        }

        assert((uintptr_t)cur_vertex_ptr % (uintptr_t)sizeof(uint32_t) ==
               0);

        uint32_t *cur_index_ptr = (uint32_t *)cur_vertex_ptr;
        int64_t cur_index_offset = 
            ((char *)cur_index_ptr - obj_pack_base) / sizeof(uint32_t);

        for (int64_t mesh_idx = 0; mesh_idx < num_meshes; mesh_idx++) {
            const SourceMesh &src_mesh = src_obj.meshes[mesh_idx];
            Mesh &dst_mesh = metadata.meshes[obj_mesh_offset + mesh_idx];

            int64_t num_indices = src_mesh.indices.size();

            dst_mesh.indexOffset = uint32_t(cur_index_offset); // FIXME
            dst_mesh.numIndices = uint32_t(num_indices);
            obj_meshdata[mesh_idx].indexOffset = uint32_t(cur_index_offset);

            memcpy(cur_index_ptr, src_mesh.indices.data(),
                   num_indices * sizeof(uint32_t));

            cur_index_ptr += num_indices;
            cur_index_offset += num_indices;
        }

        int64_t total_obj_bytes = utils::roundUpPow2(
            (char *)cur_index_ptr - obj_pack_base,
            sizeof(shader::PackedVertex));

        static_assert(utils::isPower2(sizeof(shader::PackedVertex)));

        pack_offset += total_obj_bytes;
        obj_mesh_offset += num_meshes;
    }
}

Assets AssetManager::load(const DeviceState &dev,
                          MemoryAllocator &mem,
                          const AssetMetadata &metadata,
                          HostBuffer &&staging_buffer)
{
    LocalBuffer geo_buffer =
        mem.makeLocalBuffer(metadata.numGPUDataBytes, true).value();

    VkQueue tmp_queue_hdl = makeQueue(dev, dev.computeQF, 1);
    QueueState tmp_queue(tmp_queue_hdl, false);
    VkFence tmp_fence = makeFence(dev);

    VkCommandPool tmp_cmd_pool = makeCmdPool(dev, dev.computeQF);
    VkCommandBuffer tmp_cmd_buffer = makeCmdBuffer(dev, tmp_cmd_pool);

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    REQ_VK(dev.dt.beginCommandBuffer(tmp_cmd_buffer, &begin_info));

    VkBufferCopy geo_copy_info {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = metadata.numGPUDataBytes,
    };
    dev.dt.cmdCopyBuffer(tmp_cmd_buffer, staging_buffer.buffer,
                         geo_buffer.buffer, 1, &geo_copy_info);

    REQ_VK(dev.dt.endCommandBuffer(tmp_cmd_buffer));

    VkSubmitInfo copy_submit {};
    copy_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    copy_submit.waitSemaphoreCount = 0;
    copy_submit.pWaitSemaphores = nullptr;
    copy_submit.pWaitDstStageMask = nullptr;
    copy_submit.commandBufferCount = 1;
    copy_submit.pCommandBuffers = &tmp_cmd_buffer;

    tmp_queue.submit(dev, 1, &copy_submit, tmp_fence);
    waitForFenceInfinitely(dev, tmp_fence);
    resetFence(dev, tmp_fence);

    VkDeviceAddress geo_dev_addr = getDevAddr(dev, geo_buffer.buffer);

    REQ_VK(dev.dt.resetCommandPool(dev.hdl, tmp_cmd_pool, 0));

    REQ_VK(dev.dt.beginCommandBuffer(tmp_cmd_buffer, &begin_info));

    auto blas_build = makeBLASes(
        dev, mem, metadata, geo_dev_addr, tmp_cmd_buffer);

    REQ_VK(dev.dt.endCommandBuffer(tmp_cmd_buffer));

    tmp_queue.submit(dev, 1, &copy_submit, tmp_fence);
    waitForFenceInfinitely(dev, tmp_fence);
    resetFence(dev, tmp_fence);

    int64_t num_objects = metadata.objects.size();
    int64_t base_obj_offset = freeObjectOffset;

    uint64_t *blas_addrs_ptr =
        (uint64_t *)addrBufferStaging.ptr + base_obj_offset;
    shader::ObjectData *object_data_ptr = (shader::ObjectData *)(
        (uint64_t *)addrBufferStaging.ptr + maxObjects) + base_obj_offset;

    for (int64_t i = 0; i < num_objects; i++) {
        blas_addrs_ptr[i] = blas_build->blases.accelStructs[i].devAddr;
        object_data_ptr[i].geoAddr =
            geo_dev_addr + metadata.objectOffsets[i];
    }

    freeObjectOffset += num_objects;

    addrBufferStaging.flush(dev);

    REQ_VK(dev.dt.resetCommandPool(dev.hdl, tmp_cmd_pool, 0));
    REQ_VK(dev.dt.beginCommandBuffer(tmp_cmd_buffer, &begin_info));

    std::array<VkBufferCopy, 2> addrCopies;
    uint64_t blas_addr_start_offset =
        (char *)blas_addrs_ptr - (char *)addrBufferStaging.ptr;
    addrCopies[0].srcOffset = blas_addr_start_offset;
    addrCopies[0].dstOffset = blas_addr_start_offset;
    addrCopies[0].size = sizeof(uint64_t) * num_objects;

    uint64_t object_data_start_offset =
        (char *)object_data_ptr - (char *)addrBufferStaging.ptr;
    addrCopies[1].srcOffset = object_data_start_offset;
    addrCopies[1].dstOffset = object_data_start_offset;
    addrCopies[1].size = sizeof(shader::ObjectData) * num_objects;

    dev.dt.cmdCopyBuffer(tmp_cmd_buffer, addrBufferStaging.buffer,
                         addrBuffer.buf.buffer, addrCopies.size(),
                         addrCopies.data());

    REQ_VK(dev.dt.endCommandBuffer(tmp_cmd_buffer));

    tmp_queue.submit(dev, 1, &copy_submit, tmp_fence);
    waitForFenceInfinitely(dev, tmp_fence);
    resetFence(dev, tmp_fence);

    return Assets {
        std::move(geo_buffer),
        std::move(blas_build->blases),
        base_obj_offset,
    };
}

Assets AssetManager::loadCube(const DeviceState &dev,
                              MemoryAllocator &mem)
{
    std::vector<SourceVertex> vertices {
        {
            Vector3 { -1, -1, -1, },
            Vector3 {},
            Vector4 {},
            Vector2 {},
        },
        {
            Vector3 {  1, -1, -1, },
            Vector3 {},
            Vector4 {},
            Vector2 {},
        },
        {
            Vector3 {  1,  1, -1, },
            Vector3 {},
            Vector4 {},
            Vector2 {},
        },
        {
            Vector3 { -1,  1, -1, },
            Vector3 {},
            Vector4 {},
            Vector2 {},
        },
        {
            Vector3 { -1, -1,  1, },
            Vector3 {},
            Vector4 {},
            Vector2 {},
        },
        {
            Vector3 {  1, -1,  1, },
            Vector3 {},
            Vector4 {},
            Vector2 {},
        },
        {
            Vector3 {  1,  1,  1, },
            Vector3 {},
            Vector4 {},
            Vector2 {},
        },
        {
            Vector3 { -1,  1,  1, },
            Vector3 {},
            Vector4 {},
            Vector2 {},
        },
    };

    std::vector<uint32_t> indices {
        0, 1, 3, 3, 1, 2,
        1, 5, 2, 2, 5, 6,
        5, 4, 6, 6, 4, 7,
        4, 0, 7, 7, 0, 3,
        3, 2, 7, 7, 2, 6,
        4, 5, 0, 0, 5, 1,
    };

    std::vector<SourceMesh> meshes {
        {
            { vertices.data(), CountT(vertices.size()) },
            { indices.data(), CountT(indices.size()) },
        },
    };

    std::vector<SourceObject> objects {
        { 
            { meshes.data(), CountT(meshes.size()) },
        },
    };

    Span<const SourceObject> objs_view(objects.data(), objects.size());

    AssetMetadata metadata = *prepareMetadata(objs_view);

    HostBuffer staging_buffer =
        mem.makeStagingBuffer(metadata.numGPUDataBytes);

    packAssets(staging_buffer.ptr, metadata, objs_view);

    staging_buffer.flush(dev);

    return load(dev, mem, metadata, std::move(staging_buffer));
}


TLASData TLASData::setup(const DeviceState &dev,
                         const GPURunUtil &gpu_run,
                         int cuda_gpu_id,
                         MemoryAllocator &mem,
                         int64_t num_worlds,
                         uint32_t max_num_instances)
{
    HeapArray<VkAccelerationStructureGeometryKHR> geometry_infos(num_worlds);
    HeapArray<VkAccelerationStructureBuildGeometryInfoKHR> build_infos(
        num_worlds);

    for (int64_t i = 0; i < num_worlds; i++) {
        VkAccelerationStructureGeometryKHR &geo_info = geometry_infos[i];
        geo_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geo_info.pNext = nullptr;
        geo_info.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geo_info.flags = 0;
        auto &geo_instances = geo_info.geometry.instances;
        geo_instances.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geo_instances.pNext = nullptr;
        geo_instances.arrayOfPointers = false;
        geo_instances.data.deviceAddress = 0;

        VkAccelerationStructureBuildGeometryInfoKHR &build_info = 
            build_infos[i];
        build_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        build_info.pNext = nullptr;
        build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        build_info.flags =
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build_info.srcAccelerationStructure = VK_NULL_HANDLE;
        build_info.dstAccelerationStructure = VK_NULL_HANDLE;
        build_info.geometryCount = 1;
        build_info.pGeometries = &geo_info;
        build_info.ppGeometries = nullptr;
        build_info.scratchData.deviceAddress = 0;
    }

    VkAccelerationStructureBuildSizesInfoKHR size_info;
    size_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    size_info.pNext = nullptr;

    dev.dt.getAccelerationStructureBuildSizesKHR(dev.hdl,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_infos[0], &max_num_instances, &size_info);

    // Acceleration structure storage needs to be on 256 byte boundaries
    uint64_t num_bytes_per_as = 
        utils::roundUpPow2(size_info.accelerationStructureSize, 256);

    uint64_t total_as_bytes = num_worlds * (
        num_bytes_per_as + size_info.buildScratchSize);

    DedicatedBuffer as_storage = mem.makeDedicatedBuffer(total_as_bytes, true);
    VkDeviceAddress as_storage_addr = getDevAddr(dev, as_storage.buf.buffer);
    VkDeviceAddress cur_as_scratch_addr =
        as_storage_addr + num_worlds * num_bytes_per_as;

    uint64_t initial_instance_storage_bytes =
        num_worlds * max_num_instances * sizeof(AccelStructInstance);

    DedicatedBuffer instance_storage = mem.makeDedicatedBuffer(
        initial_instance_storage_bytes, true);
    CudaImportedBuffer instance_storage_cuda(dev, cuda_gpu_id,
        instance_storage.mem, initial_instance_storage_bytes);

    VkDeviceAddress instance_storage_base_addr =
        getDevAddr(dev, instance_storage.buf.buffer);

    uint64_t num_instance_addrs_bytes =
        sizeof(AccelStructInstance *) * num_worlds;

    HostBuffer instance_addrs_staging =
        mem.makeStagingBuffer(num_instance_addrs_bytes);
    auto instance_addrs_staging_ptr =
        (AccelStructInstance **)instance_addrs_staging.ptr;

    DedicatedBuffer instance_addrs_storage =
        mem.makeDedicatedBuffer(num_instance_addrs_bytes);
    CudaImportedBuffer instance_addrs_storage_cuda(dev, cuda_gpu_id,
        instance_addrs_storage.mem, num_instance_addrs_bytes);

    HeapArray<VkAccelerationStructureKHR> hdls(num_worlds);
    HeapArray<uint32_t> max_instances(num_worlds);

    HeapArray<VkAccelerationStructureBuildRangeInfoKHR> range_infos(
        num_worlds);
    HeapArray<VkAccelerationStructureBuildRangeInfoKHR *> range_info_ptrs(
        num_worlds);

    VkDeviceSize cur_as_storage_offset = 0;
    VkDeviceSize cur_instance_storage_offset = 0;
    for (int64_t i = 0; i < num_worlds; i++) {
        // Prepopulate geometry info with device address
        geometry_infos[i].geometry.instances.data.deviceAddress =
            instance_storage_base_addr + cur_instance_storage_offset;
        instance_addrs_staging_ptr[i] = (AccelStructInstance *)
            ((char *)instance_storage_cuda.getDevicePointer() +
            cur_instance_storage_offset);
        max_instances[i] = max_num_instances;

        cur_instance_storage_offset +=
            sizeof(AccelStructInstance) * max_num_instances;

        VkAccelerationStructureCreateInfoKHR create_info;
        create_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        create_info.pNext = nullptr;
        create_info.createFlags = 0;
        create_info.buffer = as_storage.buf.buffer;
        create_info.offset = cur_as_storage_offset;
        create_info.size = size_info.accelerationStructureSize;
        create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        create_info.deviceAddress = 0;

        cur_as_storage_offset += num_bytes_per_as;

        REQ_VK(dev.dt.createAccelerationStructureKHR(dev.hdl, &create_info,
                                                     nullptr, &hdls[i]));

        auto &build_info = build_infos[i];
        build_info.dstAccelerationStructure = hdls[i];
        build_info.scratchData.deviceAddress = cur_as_scratch_addr;

        cur_as_scratch_addr += size_info.buildScratchSize;

        VkAccelerationStructureBuildRangeInfoKHR &range_info = range_infos[i];
        range_info.primitiveCount = -1;
        range_info.primitiveOffset = 0;
        range_info.firstVertex = 0;
        range_info.transformOffset = 0;

        range_info_ptrs[i] = &range_info;
    }

    instance_addrs_staging.flush(dev);

    gpu_run.begin(dev);

    VkBufferCopy addrs_copy_info {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = num_instance_addrs_bytes,
    };

    dev.dt.cmdCopyBuffer(gpu_run.cmd, instance_addrs_staging.buffer,
                         instance_addrs_storage.buf.buffer, 1,
                         &addrs_copy_info);

    gpu_run.submit(dev);

    return TLASData {
        std::move(as_storage),
        std::move(instance_storage),
        std::move(instance_storage_cuda),
        std::move(instance_addrs_storage),
        std::move(instance_addrs_storage_cuda),
        std::move(hdls),
        std::move(max_instances),
        std::move(geometry_infos),
        std::move(build_infos),
        std::move(range_infos),
        std::move(range_info_ptrs),
    };
}

void TLASData::build(const DeviceState &dev,
                     const uint32_t *num_instances_per_world,
                     VkCommandBuffer build_cmd)
{
    const int64_t num_worlds = hdls.size();
    for (int64_t i = 0; i < num_worlds; i++) {
        uint32_t num_instances = num_instances_per_world[i];
        assert(num_instances <= maxInstances[i]);
        auto &range_info = rangeInfos[i];
        range_info.primitiveCount = num_instances;
    }

    dev.dt.cmdBuildAccelerationStructuresKHR(build_cmd, 1, buildInfos.data(),
                                             rangeInfoPtrs.data());
}

void TLASData::free(const DeviceState &dev)
{
    const int64_t num_worlds = hdls.size();
    for (int64_t i = 0; i < num_worlds; i++) {
        dev.dt.destroyAccelerationStructureKHR(dev.hdl, hdls[i], nullptr);
    }
}

#if 0
void TLAS::build(const DeviceState &dev,
                 MemoryAllocator &alloc,
                 VkDeviceAddress instance_buffer_addr,
                 uint32_t num_instances,
                 const BLASData &blases,
                 VkCommandBuffer build_cmd)
{
    VkAccelerationStructureInstanceKHR *accel_insts =
        reinterpret_cast<VkAccelerationStructureInstanceKHR  *>(
            buildStorage->ptr);

    for (int inst_idx = 0; inst_idx < new_num_instances; inst_idx++) {
        const ObjectInstance &inst = instances[inst_idx];
        const InstanceTransform &txfm = instance_transforms[inst_idx];

        VkAccelerationStructureInstanceKHR &inst_info =
            accel_insts[inst_idx];

        memcpy(&inst_info.transform,
               glm::value_ptr(glm::transpose(txfm.mat)),
               sizeof(VkTransformMatrixKHR));

        if (instance_flags[inst_idx] & InstanceFlags::Transparent) {
            inst_info.mask = 2;
        } else {
            inst_info.mask = 1;
        }
        inst_info.instanceCustomIndex = inst.materialOffset;
        inst_info.instanceShaderBindingTableRecordOffset = 
            objects[inst.objectIndex].meshIndex;
        inst_info.flags = 0;
        inst_info.accelerationStructureReference =
            blases.accelStructs[inst.objectIndex].devAddr;
    }

    buildStorage->flush(dev);
#endif


}
}
}
