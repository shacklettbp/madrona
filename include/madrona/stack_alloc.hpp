#pragma once

#include <madrona/types.hpp>
#include <madrona/utils.hpp>

namespace madrona {

class StackAlloc {
public:
    struct Frame {
        void *ptr;
    };

    StackAlloc(CountT chunk_size = 32768);
    StackAlloc(const StackAlloc &) = delete;
    ~StackAlloc();

    inline Frame push();
    void pop(Frame frame);

    inline void * alloc(CountT num_bytes, CountT alignment);

    template <typename T>
    T * alloc();

    template <typename T>
    T * allocN(CountT num_elems);

private:
    struct ChunkMetadata {
        ChunkMetadata *next;
    };

    [[noreturn]] void allocTooLarge();

    static char * newChunk(CountT num_bytes, CountT alignment);

    char *first_chunk_;
    char *cur_chunk_;
    uint32_t chunk_offset_;
    uint32_t chunk_size_;
};

}

#include "stack_alloc.inl"
