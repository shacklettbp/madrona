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
#endif

#include <algorithm>

namespace madrona {

struct VirtualRegion::Init {
    char *const base;
    uint64_t chunkShift;
    uint64_t totalSize;
    uint64_t initChunks;

    static inline Init make(uint64_t max_bytes, uint64_t chunk_shift,
                            uint64_t alignment, uint64_t init_chunks)
    {
        uint32_t page_size =
#if defined(__linux__) or defined(__APPLE__)
            sysconf(_SC_PAGESIZE);
#elif _WIN32
            STATIC_UNIMPLEMENTED();
#else
            STATIC_UNIMPLEMENTED();
#endif
    
        if (chunk_shift == 0) {
            chunk_shift = utils::int32Log2(page_size);
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
    
        // mmap guarantees at least alignment to page boundaries already
        if (alignment <= page_size) {
            alignment = 1;
        }
    
        if (alignment > 1 && alignment % page_size != 0) {
            FATAL("VirtualRegion alignment must be multiple of page size (or 1)");
        }
    
        uint64_t max_chunks = utils::divideRoundUp(max_bytes, chunk_size);
    
#if defined(__linux__) or defined(__APPLE__)
        uint64_t overalign_size = (max_chunks << chunk_shift) + alignment - 1;
    
#ifdef __linux__
        constexpr int mmap_init_flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#elif defined(__APPLE__)
        constexpr int mmap_init_flags = MAP_PRIVATE | MAP_ANON;
#endif
    
        void *base_attempt =
            mmap(nullptr, overalign_size, PROT_NONE, mmap_init_flags, -1, 0);
    
        if (base_attempt == MAP_FAILED) [[unlikely]] {
            FATAL("Failed to allocate %lu bytes of virtual address space\n",
                  overalign_size);
        }
    
        char *base = (char *)utils::roundUp((uintptr_t)base_attempt, (uintptr_t)alignment);
        uint64_t extra_amount = base - (char *)base_attempt;
    
        if (extra_amount > 0) {
            munmap(base_attempt, extra_amount);
        }
    
        uint64_t total_size = overalign_size - extra_amount;
    
#elif _WIN32
        STATIC_UNIMPLEMENTED();
#else
        STATIC_UNIMPLEMENTED();
#endif

        return Init {
            .base = base,
            .chunkShift = chunk_shift,
            .totalSize = total_size,
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
      chunk_shift_(init.chunkShift),
      total_size_(init.totalSize)
{
    if (init.initChunks > 0) {
        commit(0, init.initChunks);
    }
}

VirtualRegion::~VirtualRegion()
{
#if defined(__linux__) or defined(__APPLE__)
    munmap(base_, total_size_);
#elif defined(_WIN32)
    STATIC_UNIMPLEMENTED();
#else
    STATIC_UNIMPLEMENTED();
#endif
}

void VirtualRegion::commit(uint64_t start_chunk, uint64_t num_chunks)
{
#if defined(__linux__) or defined(__APPLE__)
    void *res = mmap(base_ + (start_chunk << chunk_shift_),
        num_chunks << chunk_shift_, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);

    if (res == MAP_FAILED) {
        FATAL("Failed to commit %lu chunks for VirtualRegion", num_chunks);
    }
#elif defined(_WIN32)
    STATIC_UNIMPLEMENTED();
#else
    STATIC_UNIMPLEMENTED();
#endif
}

void VirtualRegion::decommit(uint64_t start_chunk, uint64_t num_chunks)
{
#if defined(__linux__) or defined(__APPLE__)
    // FIXME MADV_FREE instead
    int res = madvise(base_ + (start_chunk << chunk_shift_),
                      num_chunks << chunk_shift_, 
#ifdef MADRONA_LINUX
                      MADV_REMOVE);
#elif defined(MADRONA_MACOS)
                      MADV_FREE);
#endif


    if (res != 0) {
        FATAL("Failed to decommit %lu chunks for VirtualRegion", num_chunks);
    }
#elif defined(_WIN32)
    STATIC_UNIMPLEMENTED();
#else
    STATIC_UNIMPLEMENTED();
#endif
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
        region_.commit(committed_chunks_, 1);
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
        region_.decommit(committed_chunks_, 1);

        committed_items_ = computeCommittedItems(committed_chunks_,
            bytes_per_item_, start_offset_, region_); 
    }
}

}
