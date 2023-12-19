#include "mw_gpu/const.hpp"

namespace madrona {
namespace mwGPU {

ChunkAllocator & ChunkAllocator::get()
{
    return *(ChunkAllocator *)GPUImplConsts::get().chunkAllocatorAddr;
}

void * ChunkAllocator::base()
{
    return GPUImplConsts::get().chunkBaseAddr;
}

void * ChunkAllocator::chunkPtr(uint32_t chunk_idx)
{
    return (char *)ChunkAllocator::base() + (uint64_t)chunkSize * (uint64_t)i;
}

uint32_t ChunkAllocator::allocChunk(Cache &cache)
{
    if (cache.cachedIdx != ~0u) {
        uint32_t chunk_idx = cache.cache_idx;
        cache.cache_idx = ~0u;
        return chunk_idx;
    }

    FreeHead cur_head = free_head_.load_acquire();
    FreeHead new_head;
    Chunk *node;

    do {
        if (cur_head.head == ~0u) {
            break;
        }

        new_head.gen = cur_head.gen + 1;
        node = (Node *)ChunkAllocator::chunkPtr(cur_head.head);
        new_head.head = node->next;
    } while (!free_head_.compare_exchange_weak<sync::release, sync::acquire>(
        cur_head, new_head));

    return cur_head.head;
}

void ChunkAllocator::freeChunk(Cache &cache, uint32_t chunk_idx)
{
    if (cache.cachedIdx == ~0u) {
        cache.cachedIdx = chunk_idx;

        return;
    }

    FreeHead cur_head = free_head_.load_relaxed();
    FreeHead new_head;
    new_head.head = chunk_idx;
    Node &new_node = *(Node *)ChunkAllocator::chunkPtr(chunk_idx);

    do {
        new_head.gen = cur_head.gen + 1;
        new_node.next = cur_head.head;
    } while (!free_head_.compare_exchange_weak<sync::release, sync::relaxed>(
        cur_head, new_head));
}

}
}
