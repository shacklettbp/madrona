#pragma once

#include <cstdint>
#include <atomic>

namespace madrona {
namespace mwGPU {


struct RefChunk {
    RefChunk(uint32_t chunk);
    void *addr;
    uint32_t freeBytes;

    void acquireRef(uint32_t num_release);
    bool releaseRef(uint32_t num_release);
};

class ChunkAllocator {
public:
    ChunkAllocator(uint32_t num_chunks);

    struct Cache {
        uint32_t cachedIdx = ~0u;
    };

    static inline ChunkAllocator & get();
    static inline void *base();
    static inline void *chunkPtr(uint32_t chunk_idx);

    inline uint32_t allocChunk(Cache &cache);
    inline void freeChunk(Cache &cache, uint32_t chunk_idx);

    static constexpr uint32_t chunkSize = 4096;
private:
    struct Node {
        uint32_t next;
    };

    struct alignas(std::atomic_uint64_t) FreeHead {
        uint32_t gen;
        uint32_t head;
    };

    std::atomic<FreeHead> free_head_;
    static_assert(decltype(free_head_)::is_always_lock_free);
};

}
}
