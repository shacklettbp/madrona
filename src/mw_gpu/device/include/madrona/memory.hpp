#pragma once

#ifdef MADRONA_GPU_MODE

#include <cstdint>
#include <cuda/atomic>

#include <madrona/sync.hpp>

#include "mw_gpu/const.hpp"

#endif

namespace madrona {

inline void * rawAlloc(uint64_t num_bytes)
{
    assert(num_bytes != 0);
    void *ptr = malloc(num_bytes);
    assert(ptr != nullptr);

    return ptr;
}

inline void rawDealloc(void *ptr)
{
    free(ptr);
}

class DefaultAlloc  {
public:
    inline void * alloc(uint64_t num_bytes)
    {
        return rawAlloc(num_bytes);
    }

    inline void dealloc(void *ptr)
    {
        rawDealloc(ptr);
    }
};

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

}
}

#ifdef MADRONA_GPU_MODE

namespace madrona {
namespace mwGPU {

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

namespace SharedMemStorage {
    struct alignas(64) Chunk {};

    //inline constexpr uint64_t numSMemBytes = 15872;
    // Use a bit lower than max smem here as there are a few
    // random places using their own smem allocations
    inline constexpr uint64_t numSMemBytes = 15040;
    extern __shared__ Chunk buffer[
        numSMemBytes / sizeof(Chunk)];
};

class TmpAllocator {
public:
    TmpAllocator();

    void * alloc(uint64_t num_bytes);

    inline void reset()
    {
        offset_.store(0, std::memory_order_relaxed);
    }

    static inline TmpAllocator & get()
    {
        return *(TmpAllocator *)
            mwGPU::GPUImplConsts::get().tmpAllocatorAddr;
    }

private:
    void *base_;
    std::atomic_uint64_t offset_;
    uint64_t num_mapped_bytes_;
    utils::SpinLock grow_lock_;
};

}

}

#endif
