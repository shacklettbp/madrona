/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/virtual.hpp>
#include <madrona/macros.hpp>
#include <madrona/sync.hpp>
#include <madrona/types.hpp>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace madrona {

// FIXME get rid of these eventually:

inline void * rawAlloc(size_t num_bytes)
{
    return malloc(num_bytes);
}

inline void rawDealloc(void *ptr)
{
    free(ptr);
}

inline void * rawAllocAligned(size_t num_bytes, size_t alignment);
inline void rawDeallocAligned(void *ptr);

class OSAlloc {
    struct Block;
public:
    class Cache {
    public:
        Cache();
        Cache(const Cache &) = delete;
        Cache(Cache &&o);

    private:
        uint32_t cache_head_;
        uint32_t num_cache_blocks_;

    friend class OSAlloc;
    };

    OSAlloc();

    void * getChunk(Cache &cache);
    void freeChunk(Cache &cache, void *ptr);

    static constexpr uint64_t chunkSize() { return chunk_size_; }
    static constexpr uint64_t chunkShift() { return chunk_shift_; }

private:
    struct alignas(AtomicU64) FreeHead {
        uint32_t gen;
        uint32_t head;
    };

    // 256 KiB chunks
    static constexpr uint64_t chunk_shift_ = 12;
    static constexpr uint64_t chunk_size_ = 1_u64 << chunk_shift_;

    struct Block {
        union {
            uint32_t nextFree;
            char data[chunk_size_];
        };
    };

    inline Block * getBlock(uint32_t idx);

    VirtualRegion region_;
    uint64_t mapped_chunks_;
    alignas(MADRONA_CACHE_LINE) Atomic<FreeHead> free_head_;
    SpinLock expand_lock_;
};

// Virtual Adapter
class PolyAlloc {
public:
    inline PolyAlloc(void *state,
                     void *(*alloc_ptr)(void *, size_t),
                     void (*dealloc_ptr)(void *, void *));
    inline void * alloc(size_t num_bytes);
    inline void dealloc(void *ptr);

private:
    void *state_;
    void *(*alloc_ptr_)(void *, size_t);
    void (*dealloc_ptr_)(void *, void *);
};

class AllocContext;

class AllocScope {
public:
    AllocScope(const AllocScope &) = delete;
    ~AllocScope();

private:
    AllocScope(const PolyAlloc &alloc, AllocScope *parent,
               AllocContext *ctx);

    PolyAlloc cur_alloc_;
    AllocScope *parent_;
    AllocContext *ctx_;

friend class AllocContext;
};

class AllocContext {
public:
    AllocContext();
    AllocContext(const AllocContext &) = delete;
    
    // Allocation functions
    inline void * alloc(size_t num_bytes);
    inline void dealloc(void *ptr);

    template <typename T, typename ...Args>
    inline T * make(Args &&...args);

    template <typename T>
    inline void destroy(T *);

    // Scoped helpers
    template <typename A>
    inline AllocScope scope(A &alloc);

    template <typename A, typename Fn, typename ...Args>
    auto with(A &alloc, Fn &&fn, Args &&...args) ->
        decltype(Fn(args...));

    inline const PolyAlloc &getAlloc() { return cur_alloc_; }

private:
    PolyAlloc cur_alloc_;
    AllocScope *cur_scope_;

friend class AllocScope;
};

// Allocator base class, CRTP for polymorphic adapter
template <typename A>
class Allocator {
public:
    inline PolyAlloc getPoly();

private:
    static inline void * allocStatic(void *state, size_t num_bytes);
    static inline void deallocStatic(void *state, void *ptr);
};

class DefaultAlloc : public Allocator<DefaultAlloc> {
public:
    inline void * alloc(size_t num_bytes);
    inline void dealloc(void *ptr);
};

// FIXME:
using InitAlloc = DefaultAlloc;
using TmpAlloc = DefaultAlloc;

}

#include "memory.inl"
