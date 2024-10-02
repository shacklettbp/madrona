#pragma once

#include <madrona/types.hpp>
#include <madrona/utils.hpp>

namespace madrona {

struct AllocFrame {
    void *ptr;
};

class StackAlloc {
public:
    StackAlloc(CountT chunk_size = 32768);
    StackAlloc(const StackAlloc &) = delete;
    inline StackAlloc(StackAlloc &&o);
    inline ~StackAlloc();

    StackAlloc & operator=(const StackAlloc &) = delete;
    inline StackAlloc & operator=(StackAlloc &&o);

    inline AllocFrame push();
    void pop(AllocFrame frame);

    void release();

    inline void * alloc(CountT num_bytes, CountT alignment);

    template <typename T>
    T * alloc();

    template <typename T>
    T * allocN(CountT num_elems);

private:
    struct ChunkMetadata {
        ChunkMetadata *next;
    };

    static char * newChunk(CountT num_bytes, CountT alignment);

    char *first_chunk_;
    char *cur_chunk_;
    uint32_t chunk_offset_;
    uint32_t chunk_size_;
};

}

#include "stack_alloc.inl"
