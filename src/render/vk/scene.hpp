#pragma once

#include <madrona/render.hpp>
#include <madrona/heap_array.hpp>

#include "core.hpp"
#include "memory.hpp"
#include "cuda_interop.hpp"

namespace madrona {
namespace render {
namespace vk {

struct Mesh {
    uint32_t vertexOffset;
    uint32_t numVertices;
    uint32_t indexOffset;
    uint32_t numIndices;
};

struct Object {
    uint32_t meshOffset;
    uint32_t numMeshes;
};

struct SourceMesh {
    Span<const shader::Vertex> vertices;
    Span<const uint32_t> indices;
};

struct SourceObject {
    Span<const SourceMesh> meshes;
};

struct AssetMetadata {
    HeapArray<Mesh> meshes;
    HeapArray<Object> objects;
    HeapArray<uint32_t> objectOffsets;
    uint32_t numGPUDataBytes;
};

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
    LocalBuffer geoBuffer;
    BLASData blases;
};

struct AssetManager {
    HostBuffer addrBufferStaging;
    DedicatedBuffer addrBuffer;
    CudaImportedBuffer addrBufferCUDA;
    int64_t freeObjectOffset;
    const int64_t maxObjects;

    AssetManager(const DeviceState &dev, MemoryAllocator &mem,
                 int cuda_gpu_id, int64_t max_objects);

    Optional<AssetMetadata> prepareMetadata(Span<const SourceObject> objects);
    void packAssets(void *dst_buf,
                    AssetMetadata &prepared,
                    Span<const SourceObject> src_objects);

    Assets load(const DeviceState &dev,
                MemoryAllocator &mem,
                const AssetMetadata &metadata,
                HostBuffer &&staged_buffer);

    Assets loadCube(const DeviceState &dev,
                    MemoryAllocator &mem);
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
