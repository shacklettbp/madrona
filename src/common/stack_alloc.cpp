#include <madrona/stack_alloc.hpp>
#include <madrona/memory.hpp>
#include <cassert>

namespace madrona {

StackAlloc::StackAlloc(CountT chunk_size)
    : first_chunk_(StackAlloc::newChunk(chunk_size, chunk_size)),
      cur_chunk_(first_chunk_),
      chunk_offset_(sizeof(ChunkMetadata)),
      chunk_size_((uint32_t)chunk_size)
{
    assert(utils::isPower2((uint64_t)chunk_size));
}

StackAlloc::~StackAlloc()
{
    auto *metadata = (ChunkMetadata *)first_chunk_;
    while (metadata != nullptr) {
        auto *next = metadata->next;
        rawDeallocAligned(metadata);
        metadata = next;
    }
}

void StackAlloc::pop(Frame frame)
{
    uintptr_t mask = chunk_size_ - 1;
    uintptr_t cur_offset = (uintptr_t)frame.ptr & mask;

    void *chunk_start = (char *)frame.ptr - cur_offset;
    auto *metadata = (ChunkMetadata *)chunk_start;

    ChunkMetadata *free_chunk = metadata->next;
    while (free_chunk != nullptr) {
        ChunkMetadata *next = free_chunk->next;
        rawDeallocAligned(free_chunk);
        free_chunk = next;
    }

    metadata->next = nullptr;

    cur_chunk_ = (char *)metadata;
    chunk_offset_ = (uint32_t)cur_offset;
}

[[noreturn]] void StackAlloc::allocTooLarge()
{
    FATAL("StackAlloc cannot satisfy allocation larger than chunk size");
}

char * StackAlloc::newChunk(CountT num_bytes, CountT alignment)
{
    // FIXME: replace malloc here
    void *new_chunk = rawAllocAligned(num_bytes, alignment);

    auto *metadata = (ChunkMetadata *)new_chunk;
    metadata->next = nullptr;

    return (char *)new_chunk;
}

}
