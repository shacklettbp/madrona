#include "scene.hpp"
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
    const vector<Mesh> &meshes,
    const vector<Object> &objects,
    uint32_t max_num_vertices,
    VkDeviceAddress vert_base,
    VkDeviceAddress index_base,
    VkCommandBuffer build_cmd)
{
    vector<VkAccelerationStructureGeometryKHR> geo_infos;
    vector<uint32_t> num_triangles;
    vector<VkAccelerationStructureBuildRangeInfoKHR> range_infos;

    geo_infos.reserve(meshes.size());
    num_triangles.reserve(meshes.size());
    range_infos.reserve(meshes.size());

    vector<VkAccelerationStructureBuildGeometryInfoKHR> build_infos;
    vector<tuple<VkDeviceSize, VkDeviceSize, VkDeviceSize>> memory_locs;

    build_infos.reserve(objects.size());
    memory_locs.reserve(objects.size());

    VkDeviceSize total_scratch_bytes = 0;
    VkDeviceSize total_accel_bytes = 0;

    for (const Object &object : objects) {
        for (int mesh_idx = 0; mesh_idx < (int)object.numMeshes; mesh_idx++) {
            const Mesh &mesh = meshes[object.meshOffset + mesh_idx];

            VkDeviceAddress vert_addr = vert_base;
            VkDeviceAddress index_addr =
                index_base + mesh.indexOffset * sizeof(uint32_t);

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
            tri_info.vertexData.deviceAddress = vert_addr;
            tri_info.vertexStride = sizeof(Vertex);
            tri_info.maxVertex = max_num_vertices;
            tri_info.indexType = VK_INDEX_TYPE_UINT32;
            tri_info.indexData.deviceAddress = index_addr;
            tri_info.transformData.deviceAddress = 0;

            geo_infos.push_back(geo_info);
            num_triangles.push_back(mesh.numTriangles);

            VkAccelerationStructureBuildRangeInfoKHR range_info;
            range_info.primitiveCount = mesh.numTriangles;
            range_info.primitiveOffset = 0;
            range_info.firstVertex = 0;
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
            &num_triangles[object.meshOffset],
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

Assets Assets::load(const DeviceState &dev,
                  MemoryAllocator &mem)
{
    std::vector<Vertex> vertices {
        Vertex {
            Vector3 { -1, -1, -1, },
            Vector3 { },
            Vector2 { },
        },
        Vertex {
            Vector3 {  1, -1, -1, },
            Vector3 { },
            Vector2 { },
        },
        Vertex {
            Vector3 {  1,  1, -1, },
            Vector3 { },
            Vector2 { },
        },
        Vertex {
            Vector3 { -1,  1, -1, },
            Vector3 { },
            Vector2 { },
        },
        Vertex {
            Vector3 { -1, -1,  1, },
            Vector3 { },
            Vector2 { },
        },
        Vertex {
            Vector3 {  1, -1,  1, },
            Vector3 { },
            Vector2 { },
        },
        Vertex {
            Vector3 {  1,  1,  1, },
            Vector3 { },
            Vector2 { },
        },
        Vertex {
            Vector3 { -1,  1,  1, },
            Vector3 { },
            Vector2 { },
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

    std::vector<Mesh> meshes {
        Mesh { 0, 0, uint32_t(indices.size() / 3), },
    };

    std::vector<Object> objects {
        Object { 0, 1 },
    };

    uint64_t num_vertex_bytes = sizeof(Vertex) * vertices.size();
    uint64_t num_index_bytes = sizeof(uint32_t) * indices.size();

    LocalBuffer vert_buffer = mem.makeLocalBuffer(num_vertex_bytes, true).value();
    LocalBuffer idx_buffer = mem.makeLocalBuffer(num_index_bytes, true).value();

    HostBuffer vert_stage_buffer = mem.makeStagingBuffer(num_vertex_bytes);
    HostBuffer idx_stage_buffer = mem.makeStagingBuffer(num_index_bytes);

    memcpy(vert_stage_buffer.ptr, vertices.data(), num_vertex_bytes);
    memcpy(idx_stage_buffer.ptr, indices.data(), num_index_bytes);

    vert_stage_buffer.flush(dev);
    idx_stage_buffer.flush(dev);

    VkQueue tmp_queue_hdl = makeQueue(dev, dev.computeQF, 1);
    QueueState tmp_queue(tmp_queue_hdl, false);
    VkFence tmp_fence = makeFence(dev);

    VkCommandPool tmp_cmd_pool = makeCmdPool(dev, dev.computeQF);
    VkCommandBuffer tmp_cmd_buffer = makeCmdBuffer(dev, tmp_cmd_pool);

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    REQ_VK(dev.dt.beginCommandBuffer(tmp_cmd_buffer, &begin_info));

    VkBufferCopy vert_copy_info {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = num_vertex_bytes,
    };
    dev.dt.cmdCopyBuffer(tmp_cmd_buffer, vert_stage_buffer.buffer,
                         vert_buffer.buffer, 1, &vert_copy_info);

    VkBufferCopy idx_copy_info {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = num_index_bytes,
    };
    dev.dt.cmdCopyBuffer(tmp_cmd_buffer, idx_stage_buffer.buffer,
                         idx_buffer.buffer, 1, &idx_copy_info);

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

    VkBufferDeviceAddressInfo vert_addr_info;
    vert_addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    vert_addr_info.pNext = nullptr;
    vert_addr_info.buffer = vert_buffer.buffer;
    VkDeviceAddress vert_dev_addr =
        dev.dt.getBufferDeviceAddress(dev.hdl, &vert_addr_info);

    VkBufferDeviceAddressInfo idx_addr_info;
    idx_addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    idx_addr_info.pNext = nullptr;
    idx_addr_info.buffer = idx_buffer.buffer;
    VkDeviceAddress idx_dev_addr =
        dev.dt.getBufferDeviceAddress(dev.hdl, &idx_addr_info);

    REQ_VK(dev.dt.resetCommandPool(dev.hdl, tmp_cmd_pool, 0));

    REQ_VK(dev.dt.beginCommandBuffer(tmp_cmd_buffer, &begin_info));

    auto blas_build = makeBLASes(dev, mem, meshes, objects, vertices.size(),
        vert_dev_addr, idx_dev_addr, tmp_cmd_buffer);

    REQ_VK(dev.dt.endCommandBuffer(tmp_cmd_buffer));

    tmp_queue.submit(dev, 1, &copy_submit, tmp_fence);
    waitForFenceInfinitely(dev, tmp_fence);

    return Assets {
        std::move(vert_buffer),
        std::move(idx_buffer),
        std::move(blas_build->blases),
    };
}

TLASData TLASData::setup(const DeviceState &dev,
                         MemoryAllocator &mem,
                         int64_t num_worlds,
                         uint32_t max_num_instances)
{
    HeapArray<VkAccelerationStructureGeometryKHR> geometry_infos(num_worlds);
    HeapArray<VkAccelerationStructureBuildGeometryInfoKHR> build_infos(num_worlds);

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

        VkAccelerationStructureBuildGeometryInfoKHR &build_info = build_infos[i];
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

    uint64_t total_bytes = num_worlds * (size_info.buildScratchSize +
                                         size_info.accelerationStructureSize);
    LocalBuffer storage_buf = mem.makeLocalBuffer(total_bytes, true).value();

    VkBufferDeviceAddressInfoKHR storage_addr_info;
    storage_addr_info.sType =
        VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    storage_addr_info.pNext = nullptr;
    storage_addr_info.buffer = storage_buf.buffer;
    VkDeviceAddress storage_base = dev.dt.getBufferDeviceAddress(dev.hdl,
        &storage_addr_info);

    HeapArray<VkAccelerationStructureKHR> hdls(num_worlds);
    HeapArray<VkDeviceAddress> tlas_addrs(num_worlds);
    HeapArray<VkAccelerationStructureBuildRangeInfoKHR> range_infos(num_worlds);
    HeapArray<VkAccelerationStructureBuildRangeInfoKHR *> range_info_ptrs(num_worlds);

    VkDeviceSize cur_storage_offset = 0;
    for (int64_t i = 0; i < num_worlds; i++) {
        VkAccelerationStructureCreateInfoKHR create_info;
        create_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        create_info.pNext = nullptr;
        create_info.createFlags = 0;
        create_info.buffer = storage_buf.buffer;
        create_info.offset = cur_storage_offset;
        create_info.size = size_info.accelerationStructureSize;
        create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        create_info.deviceAddress = 0;

        cur_storage_offset += size_info.accelerationStructureSize;

        REQ_VK(dev.dt.createAccelerationStructureKHR(dev.hdl, &create_info,
                                                     nullptr, &hdls[i]));

        VkAccelerationStructureDeviceAddressInfoKHR accel_addr_info;
        accel_addr_info.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        accel_addr_info.pNext = nullptr;
        accel_addr_info.accelerationStructure = hdls[i];

        tlas_addrs[i] = dev.dt.getAccelerationStructureDeviceAddressKHR(
            dev.hdl, &accel_addr_info);

        auto &build_info = build_infos[i];
        build_info.dstAccelerationStructure = hdls[i];
        build_info.scratchData.deviceAddress = storage_base + cur_storage_offset;

        cur_storage_offset += size_info.buildScratchSize;

        VkAccelerationStructureBuildRangeInfoKHR &range_info = range_infos[i];
        range_info.primitiveCount = -1;
        range_info.primitiveOffset = 0;
        range_info.firstVertex = 0;
        range_info.transformOffset = 0;

        range_info_ptrs[i] = &range_info;
    }

    return TLASData {
        std::move(hdls),
        std::move(tlas_addrs),
        std::move(geometry_infos),
        std::move(build_infos),
        std::move(range_infos),
        std::move(range_info_ptrs),
        std::move(storage_buf),
        max_num_instances,
    };
}

void TLASData::build(const DeviceState &dev,
                     uint32_t *num_instances_per_world,
                     VkDeviceAddress instance_data_addr,
                     VkCommandBuffer build_cmd)
{
    const int64_t num_worlds = hdls.size();
    for (int64_t i = 0; i < num_worlds; i++) {
        geometryInfos[i].geometry.instances.data.deviceAddress =
            instance_data_addr;

        uint32_t num_instances = num_instances_per_world[i];
        
        assert(num_instances <= maxSupportedInstances);
        instance_data_addr += sizeof(VkAccelerationStructureInstanceKHR) *
            (uint64_t)num_instances;

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
