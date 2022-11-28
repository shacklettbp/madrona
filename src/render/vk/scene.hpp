#pragma once

#include <madrona/render.hpp>
#include <madrona/heap_array.hpp>

#include "core.hpp"
#include "memory.hpp"

namespace madrona {
namespace render {
namespace vk {

struct BLAS {
    VkAccelerationStructureKHR hdl;
    VkDeviceAddress devAddr;
};

struct BLASData {
public:
    BLASData(const DeviceState &dev, std::vector<BLAS> &&as,
             LocalBuffer &&buf);
    BLASData(const BLASData &) = delete;
    BLASData(BLASData &&o);
    ~BLASData();

    BLASData &operator=(const BLASData &) = delete;
    BLASData &operator=(BLASData &&o);

    const DeviceState *dev;
    std::vector<BLAS> accelStructs;
    LocalBuffer storage;
};

struct Assets {
    LocalBuffer vertices;
    LocalBuffer indices;
    BLASData blases;

    static Assets load(const DeviceState &dev, MemoryAllocator &mem);
};

struct TLASData {
    HeapArray<VkAccelerationStructureKHR> hdls;
    HeapArray<VkDeviceAddress> tlasAddrs;
    HeapArray<VkAccelerationStructureGeometryKHR> geometryInfos;
    HeapArray<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos;
    HeapArray<VkAccelerationStructureBuildRangeInfoKHR> rangeInfos;
    HeapArray<VkAccelerationStructureBuildRangeInfoKHR *> rangeInfoPtrs;

    LocalBuffer storage;
    uint32_t maxSupportedInstances;

    static TLASData setup(const DeviceState &dev,
                          MemoryAllocator &mem,
                          int64_t num_worlds,
                          uint32_t max_num_instances);

    void build(const DeviceState &dev,
               uint32_t *num_instances_per_world,
               VkDeviceAddress instance_data_addr_base,
               VkCommandBuffer build_cmd);

    void free(const DeviceState &dev);
};

}
}
}
