#pragma once

#include <madrona/render.hpp>
#include <madrona/heap_array.hpp>

#include "core.hpp"
#include "memory.hpp"
#include "cuda_interop.hpp"

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
    DedicatedBuffer accelStructStorage;
    DedicatedBuffer instanceStorage;
    CudaImportedBuffer instanceStorageCUDA;
    DedicatedBuffer instanceAddrsStorage;
    CudaImportedBuffer instanceAddrsStorageCUDA;

    HeapArray<VkAccelerationStructureKHR> hdls;
    HeapArray<uint32_t> maxInstances;
    HeapArray<VkAccelerationStructureGeometryKHR> geometryInfos;
    HeapArray<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos;
    HeapArray<VkAccelerationStructureBuildRangeInfoKHR> rangeInfos;
    HeapArray<VkAccelerationStructureBuildRangeInfoKHR *> rangeInfoPtrs;

    static TLASData setup(const DeviceState &dev,
                          const GPURunUtil &gpu_run,
                          int cuda_gpu_id,
                          MemoryAllocator &mem,
                          int64_t num_worlds,
                          uint32_t max_num_instances);

    void build(const DeviceState &dev,
               const uint32_t *num_instances_per_world,
               VkCommandBuffer build_cmd);

    void free(const DeviceState &dev);
};

}
}
}
