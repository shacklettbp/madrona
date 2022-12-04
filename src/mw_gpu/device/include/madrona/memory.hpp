#pragma once

#ifdef MADRONA_GPU_MODE

#include <cstdint>
#include <cuda/atomic>

#include <madrona/sync.hpp>

#include "mw_gpu/const.hpp"

#endif

namespace madrona {
namespace mwGPU {

struct HostChannel {
    enum class Op {
        Reserve,
        Map,
        Terminate,
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

    Op op;
    union {
        Reserve reserve;
        Map map;
    };

    cuda::std::atomic_uint32_t ready;
    cuda::std::atomic_uint32_t finished;
};

struct HostAllocInit {
    uint64_t pageSize;
    uint64_t allocGranularity;
    HostChannel *channel;
};

#ifdef MADRONA_GPU_MODE

class HostAllocator {
public:
    HostAllocator(HostAllocInit init);
    
    void * reserveMemory(uint64_t max_bytes, uint64_t init_num_bytes);
    void mapMemory(void *addr, uint64_t num_bytes);

    uint64_t roundUpReservation(uint64_t num_bytes);
    uint64_t roundUpAlloc(uint64_t num_bytes);

private:
    HostChannel *channel_;
    utils::SpinLock device_lock_;

    uint64_t host_page_size_;
    uint64_t alloc_granularity_;
};

inline HostAllocator * getHostAllocator()
{
    return (HostAllocator *)
        GPUImplConsts::get().hostAllocatorAddr;
}

#endif

}
}
