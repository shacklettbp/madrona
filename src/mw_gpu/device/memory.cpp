#include <madrona/memory.hpp>

namespace madrona {
namespace mwGPU {

ChunkAllocator::ChunkAllocator(uint32_t num_chunks)
    : free_head_(FreeHead {0, 0})
{
    for (uint32_t i = 0; i < num_chunks; i++) {
        auto node = (Node *)ChunkAllocator::chunkPtr(i);
        node->next = i < num_chunks - 1 ? (i + 1) : ~0u;
    }
}

}
}
