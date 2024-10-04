/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/memory.hpp>
#include <madrona/utils.hpp>

#include <algorithm>
#include <cstdlib>
#include <mutex>

#if defined(__linux__) or defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <new>

namespace madrona {
namespace {
namespace consts {
constexpr uint64_t max_memory_usage = 1024_u64 * 1024_u64 * 1024_u64;
constexpr int64_t blocks_per_cache = 16;
constexpr uint64_t virtual_map_shift = 28; // 2^28 = 256 MiB
}
}

OSAlloc::Cache::Cache()
    : cache_head_(~0_u32),
      num_cache_blocks_(0)
{}

OSAlloc::Cache::Cache(Cache &&o)
    : cache_head_(o.cache_head_),
      num_cache_blocks_(o.num_cache_blocks_)
{
    o.cache_head_ = ~0_u32;
    o.num_cache_blocks_ = 0;
}

OSAlloc::OSAlloc()
    : region_(consts::max_memory_usage, consts::virtual_map_shift, 1, 0),
      mapped_chunks_(0),
      free_head_(FreeHead {
          .gen = 0,
          .head = ~0_u32,
      })
{}

OSAlloc::Block * OSAlloc::getBlock(uint32_t idx)
{
    return &((Block *)region_.ptr())[idx];
}

void * OSAlloc::getChunk(Cache &cache)
{
    if (cache.num_cache_blocks_ != 0) {
        uint32_t cache_idx = cache.cache_head_;
        Block *cur = getBlock(cache_idx);
        cache.cache_head_ = cur->nextFree;
        --cache.num_cache_blocks_;

        return cur;
    }

    auto readNext = [&](Block *blk)
#ifdef TSAN_ENABLED
        TSAN_DISABLED 
#else
        MADRONA_ALWAYS_INLINE 
#endif
    {
        return blk->nextFree;
    };

    while (true) {
        FreeHead cur_head = free_head_.load_acquire();
        FreeHead new_head;

        Block *cur_block = nullptr;
        do {
            if (cur_head.head == ~0_u32) {
                break;
            }
            cur_block = getBlock(cur_head.head);
            new_head.gen = cur_head.gen + 1;
            new_head.head = readNext(cur_block);
        } while (!free_head_.compare_exchange_weak<
            sync::release, sync::acquire>(cur_head, new_head));

        if (cur_block != nullptr) {
            return cur_block;
        } 

        std::lock_guard lock(expand_lock_);
        cur_head = free_head_.load_relaxed();
        if (cur_head.head != ~0_u32) {
            continue;
        }

        region_.commitChunks(mapped_chunks_, 1);
        Block *new_mem = (Block *)(
            (char *)region_.ptr() + (mapped_chunks_ << consts::virtual_map_shift));

        Block *new_block = new_mem;
        uint32_t remaining_base_idx =
            uint32_t(new_block - (Block *)region_.ptr() + 1);

        constexpr int64_t num_new_blocks =
            (1_u64 << consts::virtual_map_shift) / chunk_size_ - 1;

        for (int64_t i = 0; i < num_new_blocks - 1; i++) {
            cur_block = getBlock(i + remaining_base_idx);

            cur_block->nextFree = i + remaining_base_idx + 1;
        }
        Block *last_block = getBlock(num_new_blocks - 1 + remaining_base_idx);

        cur_head = free_head_.load_relaxed();
        new_head.head = remaining_base_idx;
        do {
            new_head.gen = cur_head.gen + 1;
            last_block->nextFree = cur_head.head;
        } while (!free_head_.compare_exchange_weak<
            sync::release, sync::relaxed>(cur_head, new_head));

        return new_block;
    }
}

void OSAlloc::freeChunk(Cache &cache, void *ptr)
{
    auto free_block = (Block *)ptr;
    uint32_t free_block_idx = free_block - (Block *)region_.ptr();
    if (cache.num_cache_blocks_ < consts::blocks_per_cache) {
        free_block->nextFree = cache.cache_head_;
        cache.cache_head_ = free_block_idx;
        cache.num_cache_blocks_++;
    } else {
        free_block->nextFree = cache.cache_head_;
        Block *cur_block = free_block;

        const uint32_t num_return = consts::blocks_per_cache / 2;

        for (int64_t i = 0; i < (int64_t)num_return; i++) {
            cur_block = getBlock(cur_block->nextFree);
        }
        cache.cache_head_ = cur_block->nextFree;
        cache.num_cache_blocks_ = num_return;

        FreeHead cur_head = free_head_.load_relaxed();
        FreeHead new_head;
        new_head.head = free_block_idx;

        do {
            new_head.gen = cur_head.gen + 1;
            cur_block->nextFree = cur_head.head;
        } while (!free_head_.compare_exchange_weak<
            sync::release, sync::relaxed>(cur_head, new_head));
    }
}

AllocScope::AllocScope(const PolyAlloc &alloc, AllocScope *parent,
                       AllocContext *ctx)
    : cur_alloc_(alloc), parent_(parent), ctx_(ctx)
{
    ctx_->cur_alloc_ = alloc;
    ctx_->cur_scope_ = this;
}

AllocScope::~AllocScope()
{
    if (parent_ != nullptr) [[likely]] {
        ctx_->cur_alloc_ = parent_->cur_alloc_;
    }
    ctx_->cur_scope_ = parent_;
}

AllocContext::AllocContext()
    : cur_alloc_(nullptr, nullptr, nullptr),
      cur_scope_(nullptr)
{}

}
