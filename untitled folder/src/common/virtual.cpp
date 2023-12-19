/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/virtual.hpp>
#include <madrona/crash.hpp>
#include <madrona/utils.hpp>

#if defined(__linux__) or defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

#include <algorithm>
#include <cassert>

namespace madrona {

static void getVirtualMemProperties(uint32_t *page_size,
                                    uint32_t *alloc_granularity)
{
#if defined(__linux__) or defined(__APPLE__)
        uint32_t size = sysconf(_SC_PAGESIZE);
        *page_size = size;
        *alloc_granularity = size;
#elif defined(_WIN32)
        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);

        *page_size = sys_info.dwPageSize;
        *alloc_granularity = sys_info.dwAllocationGranularity;
#else
        STATIC_UNIMPLEMENTED();
#endif
}

struct VirtualRegion::Init {
    char *const base;
    char *const aligned;
    uint64_t chunkShift;
    uint64_t totalSize;
    uint64_t initChunks;

    static inline Init make(uint64_t max_bytes, uint64_t chunk_shift,
                            uint64_t alignment, uint64_t init_chunks)
    {
        uint32_t page_size, alloc_granularity;
        getVirtualMemProperties(&page_size, &alloc_granularity);
    
        if (chunk_shift == 0) {
            chunk_shift = (uint64_t)utils::int32Log2(page_size);
        }
    
        uint64_t chunk_size = 1ull << chunk_shift;
    
        if (chunk_size % page_size != 0) {
            FATAL("VirtualRegion chunk size must be multiple of page size");
        }
    
        // alignment == 0 is a special case where we ensure the base pointer is aligned
        // to the chunk size
        if (alignment == 0) {
            alignment = chunk_size;
        }
    
        // Region is guaranteed at least aligned to alloc granularity
        if (alignment <= alloc_granularity) {
            alignment = 1;
        }
    
        if (alignment > 1 && alignment % page_size != 0) {
            FATAL("VirtualRegion alignment must be multiple of page size (or 1)");
        }
    
        uint64_t max_chunks = utils::divideRoundUp(max_bytes, chunk_size);
        uint64_t overalign_size = (max_chunks << chunk_shift) + alignment - 1;
    
#if defined(__linux__) or defined(__APPLE__)
#ifdef __linux__
        constexpr int mmap_init_flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#elif defined(__APPLE__)
        constexpr int mmap_init_flags = MAP_PRIVATE | MAP_ANON;
#endif
    
        void *base =
            mmap(nullptr, overalign_size, PROT_NONE, mmap_init_flags, -1, 0);
    
        if (base == MAP_FAILED) [[unlikely]] {
            FATAL("Failed to allocate %lu bytes of virtual address space",
                  overalign_size);
        }
#elif _WIN32
        void *base = VirtualAlloc(nullptr, overalign_size, MEM_RESERVE,
                                  PAGE_NOACCESS);

        if (base == nullptr) [[unlikely]] {
            FATAL("Failed to allocate %lu bytes of virtual address space",
                  overalign_size);
        }
#else
        STATIC_UNIMPLEMENTED();
#endif
        uintptr_t aligned_base = utils::roundUp((uintptr_t)base,
                                                (uintptr_t)alignment);
        uint64_t extra_amount = aligned_base - (uintptr_t)base;
        assert(extra_amount <= alignment);

        void *aligned = (char *)base + extra_amount;

        return Init {
            .base = (char *)base,
            .aligned = (char *)aligned,
            .chunkShift = chunk_shift,
            .totalSize = overalign_size,
            .initChunks = init_chunks,
        };
    }
};

VirtualRegion::VirtualRegion(uint64_t max_bytes, uint64_t chunk_shift,
                             uint64_t alignment, uint64_t init_chunks)
    : VirtualRegion(Init::make(max_bytes, chunk_shift, alignment, init_chunks))
{}

VirtualRegion::VirtualRegion(Init init)
    : base_(init.base),
      aligned_(init.aligned),
      chunk_shift_(init.chunkShift),
      total_size_(init.totalSize)
{
    if (init.initChunks > 0) {
        commitChunks(0, init.initChunks);
    }
}

VirtualRegion::VirtualRegion(VirtualRegion &&o)
    : base_(o.base_),
      aligned_(o.aligned_),
      chunk_shift_(o.chunk_shift_),
      total_size_(o.total_size_)
{
    o.base_ = nullptr;
}

VirtualRegion::~VirtualRegion()
{
    if (base_ == nullptr) {
        return;
    }

#if defined(__linux__) or defined(__APPLE__)
    munmap(base_, total_size_);
#elif defined(_WIN32)
    VirtualFree(base_, 0, MEM_RELEASE);
#else
    STATIC_UNIMPLEMENTED();
#endif
}

void VirtualRegion::commitChunks(uint64_t start_chunk, uint64_t num_chunks)
{
    void *start = base_ + (start_chunk << chunk_shift_);
    uint64_t num_bytes = num_chunks << chunk_shift_;

#if defined(__linux__) or defined(__APPLE__)
    void *res = mmap(start, num_bytes, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
    bool fail = res == MAP_FAILED;
#elif defined(_WIN32)
    void *res = VirtualAlloc(start, num_bytes, MEM_COMMIT, PAGE_READWRITE);
    bool fail = res == nullptr;
#else
    STATIC_UNIMPLEMENTED();
#endif

    if (fail) [[unlikely]] {
        FATAL("Failed to commit %lu chunks for VirtualRegion", num_chunks);
    }
}

void VirtualRegion::decommitChunks(uint64_t start_chunk, uint64_t num_chunks)
{
    void *start = base_ + (start_chunk << chunk_shift_);
    uint64_t num_bytes = num_chunks << chunk_shift_;

#if defined(__linux__) or defined(__APPLE__)
    // FIXME MADV_FREE instead
    int res = madvise(start, num_bytes,
#ifdef MADRONA_LINUX
                      MADV_REMOVE);
#elif defined(MADRONA_MACOS)
                      MADV_FREE);
#endif
    bool fail = res != 0;

#elif defined(_WIN32)
    int res = VirtualFree(start, num_bytes, MEM_DECOMMIT);
    bool fail = res == 0;
#else
    STATIC_UNIMPLEMENTED();
#endif

    if (fail) {
        FATAL("Failed to decommit %lu chunks for VirtualRegion", num_chunks);
    }
}

static uint64_t computeChunkShift(uint32_t bytes_per_item)
{
    static constexpr uint64_t min_chunk_shift = 14;
    static constexpr uint64_t max_chunk_shift = 22;
    static constexpr uint64_t target_items_per_chunk = 4096;

    uint64_t target_total_bytes = target_items_per_chunk * bytes_per_item;

    uint64_t target_shift = utils::int64Log2(target_total_bytes);

    return std::clamp(target_shift, min_chunk_shift, max_chunk_shift);
}

VirtualStore::VirtualStore(uint32_t bytes_per_item,
                           uint32_t item_alignment,
                           uint32_t start_offset,
                           uint32_t max_items)
    : region_((uint64_t)bytes_per_item * (uint64_t)max_items,
              computeChunkShift(bytes_per_item), 1, 0),
      data_((void *)utils::roundUp((uintptr_t)region_.ptr() + start_offset,
                                   (uintptr_t)item_alignment)),
      bytes_per_item_(bytes_per_item),
      start_offset_((char *)data_ - (char *)region_.ptr()),
      committed_chunks_(0),
      committed_items_(0)
{
    if (start_offset_ >= region_.chunkSize()) {
        FATAL("VirtualRegion: start_offset too large");
    }
}

static uint32_t computeCommittedItems(uint32_t committed_chunks,
                                      uint32_t bytes_per_item,
                                      uint32_t offset,
                                      VirtualRegion &region)
{
    uint64_t committed_bytes = committed_chunks * region.chunkSize() - offset;
    return committed_bytes / bytes_per_item;
}

void VirtualStore::expand(uint32_t num_items)
{
    if (num_items > committed_items_) {
        region_.commitChunks(committed_chunks_, 1);
        committed_chunks_++;

        committed_items_ = computeCommittedItems(committed_chunks_,
            bytes_per_item_, start_offset_, region_); 
    }
}

void VirtualStore::shrink(uint32_t num_items)
{
    uint32_t num_excess = committed_items_ - num_items;

    if (num_excess * bytes_per_item_ > region_.chunkSize()) {
        committed_chunks_--;
        region_.decommitChunks(committed_chunks_, 1);

        committed_items_ = computeCommittedItems(committed_chunks_,
            bytes_per_item_, start_offset_, region_); 
    }
}

}
