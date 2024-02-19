#include <atomic>
#include <madrona/math.hpp>
#include <madrona/memory.hpp>

using namespace madrona;

struct alignas(16) PerspectiveCameraData {
    math::Vector3 position;
    math::Quat rotation;
    float xScale;
    float yScale;
    float zNear;
    int32_t worldIDX;
    uint32_t pad;
};

struct alignas(16) InstanceData {
    math::Vector3 position;
    math::Quat rotation;
    math::Diag3x3 scale;
    int32_t objectID;
    int32_t worldIDX;
};

// We need to be able to communicate with the memory allocator
struct HostChannel {
    enum class Op {
        Reserve,
        Map,
        Alloc,
        Terminate
    };

    struct Reserve {
        uint64_t maxBytes;
        uint64_t initNumBytes;
        void *result;
    };

    struct Map {
        void *addr;
        uint64_t numBytes;
    };

    struct Alloc {
        uint64_t numBytes;
        void *result;
    };

    Op op;

    union {
        Reserve reserve;
        Map map;
        Alloc alloc;
    };

    cuda::atomic<uint32_t, cuda::thread_scope_system> ready;
    cuda::atomic<uint32_t, cuda::thread_scope_system> finished;
};

struct InternalNode {
    uint32_t start;
    uint32_t splitIndex;
    uint32_t end;
};

// Internal data for the BVH
struct BVHInternalData {
    // These are the internal nodes. Needs to be properly allocated to
    // accomodate for the number of instances.
    InternalNode *internalNodes;
    uint32_t allocatedNodes;


    // For memory allocation purposes
    HostChannel *hostChannel;
    uint64_t pageSize;
    uint64_t granularity;
};

struct BVHParams {
    uint32_t numWorlds;
    InstanceData *instances;
    PerspectiveCameraData *views;
    int32_t *instanceOffsets;
    int32_t *instanceCounts;
    int32_t *viewOffsets;
    uint32_t *mortonCodes;

    // These are all going to be inherited from the ECS
    mwGPU::HostAllocator *hostAllocator;
    mwGPU::TmpAllocator *tmpAllocator;
};

extern "C" {
    __constant__ BVHParams bvhParams;
}

struct HostAllocInit {
    uint64_t pageSize;
    uint64_t allocGranularity;
    HostChannel *channel;
};

extern "C" __global__ void bvhInit()
{
    printf("Hello from bvhInit\n");
    printf("Got numWorlds=%u\n", bvhParams.numWorlds);
}

// For now, just use #defines for parameterizing the kernels
#define BVH_MORTON_CODES_GRAIN_SIZE 8

// Because the number of instances / views isn't going to be known when the
// CPU launches this kernel (it just gets put into a CUgraph and run every
// frame), we are just going to max out the GPU and use a persistent thread
// approach to make sure all the work gets done.

// Stages of the top-level BVH build and ray cast
// 1) Generate the internal nodes
// 2) Optimize the BVH
extern "C" __global__ void bvhAllocInternalNodes()
{
    // We need to make sure we have enough internal nodes for the initial
    // 2-wide BVH which gets constructed before the optimized tree
    uint32_t num_instances = bvhParams.instanceOffsets[bvhParams.numWorlds-1] +
                             bvhParams.instanceCounts[bvhParams.numWorlds-1];

    // For the 2-wide tree, we need about num_instances internal nodes
    uint32_t num_required_nodes = num_instances;
    uint32_t num_bytes = num_required_nodes * sizeof(InternalNode);

    auto *ptr = bvhParams.tmpAllocator->alloc(num_bytes);
    printf("From allocInternalNode: tmp allocated: %p\n", ptr);

    uint32_t *ptr_u32 = (uint32_t *)ptr;
    *ptr_u32 = 42;
}

extern "C" __global__ void bvhEntry()
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t num_worlds = bvhParams.numWorlds;
        uint32_t last_world_offset = bvhParams.instanceOffsets[num_worlds-1];
        uint32_t last_world_count = bvhParams.instanceCounts[num_worlds-1];
        uint32_t num_instances = last_world_offset + last_world_count;

        printf("There are %u total instances (hostalloc %p, tmpalloc %p)\n", 
                num_instances,
                bvhParams.hostAllocator,
                bvhParams.tmpAllocator);
    }
}
