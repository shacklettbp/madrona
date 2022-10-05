/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/impl/id_map.hpp>
#include <cassert>
#include <madrona/synch.hpp>

namespace madrona {

template <typename K, typename V, template <typename> typename StoreT>
IDMap<K, V, StoreT>::Cache::Cache()
    : free_head_(~0u),
      num_free_ids_(0),
      overflow_head_(~0u),
      num_overflow_ids_(0)
{}

template <typename K, typename V, template <typename> typename StoreT>
IDMap<K, V, StoreT>::IDMap(uint32_t init_capacity)
    : free_head_(FreeHead {
          .gen = 0,
          .head = ~0u,
      }),
      store_(init_capacity)
{
    assert(init_capacity % ids_per_cache_ == 0);

    for (int base_idx = 0; base_idx < (int)init_capacity;
         base_idx += ids_per_cache_) {
        for (int i = 0; i < (int)ids_per_cache_ - 1; i++) {
            int idx = base_idx + i;
            Node &cur = store_[idx];

            cur.gen.store(0, std::memory_order_relaxed);
            cur.freeNode.subNext = idx + 1;

            if (i == 0) {
                if (base_idx + ids_per_cache_ < init_capacity) {
                    cur.freeNode.globalNext = base_idx + ids_per_cache_;
                } else {
                    cur.freeNode.globalNext = ~0u;
                }
            } else {
                cur.freeNode.globalNext = 1;
            }
        }

        Node &last = store_[base_idx + ids_per_cache_ - 1];
        last.gen.store(0, std::memory_order_relaxed);
        last.freeNode.subNext = ~0u;
        last.freeNode.globalNext = 1;
    }

    if (init_capacity > 0) {
        free_head_.store(FreeHead {
            .gen = 0,
            .head = 0,
        }, std::memory_order_release);
    }
}

template <typename K, typename V, template <typename> typename StoreT>
K IDMap<K, V, StoreT>::acquireID(Cache &cache)
{
    auto assignCachedID = [this](uint32_t *head) {
        uint32_t new_id = *head;
        Node &node = store_[new_id];

        // globalNext is overloaded when FreeNode is in the cached
        // freelist to represent multiple contiguous IDs. Contiguous
        // blocks will never be present on the global free list by design
        // so this overloading is safe
        uint32_t num_contiguous = node.freeNode.globalNext;

        if (num_contiguous == 1) {
            *head = node.freeNode.subNext;
        } else {
            uint32_t next_free = new_id + 1;
            Node &next_node = store_[next_free];
            next_node.freeNode = FreeNode {
                .subNext = node.freeNode.subNext,
                .globalNext = num_contiguous - 1,
            };
            next_node.gen.store(0, std::memory_order_relaxed);
            *head = next_free;
        }

        return K {
            .gen = node.gen.load(std::memory_order_relaxed),
            .id = new_id,
        };
    };

    // First, check if there is a free node in the overflow cache
    if (cache.num_overflow_ids_ > 0) {
        cache.num_overflow_ids_ -= 1;
        return assignCachedID(&cache.overflow_head_);
    }

    // Next, check the main cache
    if (cache.num_free_ids_ > 0) {
        cache.num_free_ids_ -= 1;
        return assignCachedID(&cache.free_head_);
    }

    // No free IDs, refill the cache from the global free list
    FreeHead cur_head = free_head_.load(std::memory_order_acquire);
    FreeHead new_head;

    Node *cur_head_node;

    // The below do-while has a benign race when reading the globalNext
    // pointer, because globalNext may be updated by a thread that has
    // successfully popped cur_head off the list. The thread reading bad
    // data will immediately fail the CAS check regardless. An alternative
    // to suppress this would be to move globalNext out of the union and
    // switch the read to a relaxed atomic load, at the cost of 4 bytes more
    // storage per node that can't be shared with V.
    auto getGlobalNext = [](Node *cur_head) TSAN_DISABLED {
        return cur_head->freeNode.globalNext;
    };

    do {
        if (cur_head.head == ~0u) {
            break;
        }

        new_head.gen = cur_head.gen + 1;
        cur_head_node = &store_[cur_head.head];
        new_head.head = getGlobalNext(cur_head_node);
    } while (!free_head_.compare_exchange_weak(
        cur_head, new_head, std::memory_order_release,
        std::memory_order_acquire));

    uint32_t free_ids = cur_head.head;

    if (free_ids != ~0u) {
        // Assign archetype to 1 (id block of size 1) so as to not confuse
        // the cache's processing of the freelist
        cur_head_node->freeNode.globalNext = 1;

        cache.free_head_ = free_ids;
        cache.num_free_ids_ = ids_per_cache_ - 1;
        return assignCachedID(&cache.free_head_);
    }

    // No free IDs at all, expand the ID store
    uint32_t block_start = store_.expand(ids_per_cache_);

    uint32_t first_id = block_start;

    Node &assigned_node = store_[first_id];
    assigned_node.gen.store(0, std::memory_order_relaxed);

    uint32_t free_start = block_start + 1;

    Node &next_free_node = store_[free_start];
    next_free_node.freeNode = FreeNode {
        .subNext = ~0u,
        // In the cached sublist, globalNext is overloaded
        // to handle contiguous free elements
        .globalNext = ids_per_cache_ - 1,
    },
    next_free_node.gen.store(0, std::memory_order_relaxed);

    cache.free_head_ = free_start;
    cache.num_free_ids_ = ids_per_cache_ - 1;

    return K {
        .gen = 0,
        .id = first_id,
    };

}

template <typename K, typename V, template <typename> typename StoreT>
void IDMap<K, V, StoreT>::releaseID(Cache &cache, uint32_t id)
{
    Node &release_node = store_[id];
    // Avoid atomic RMW, only 1 writer
    release_node.gen.store(
        release_node.gen.load(std::memory_order_relaxed) + 1);
    release_node.freeNode.globalNext = 1;

    if (cache.num_free_ids_ < ids_per_cache_) {
        release_node.freeNode.subNext = cache.free_head_;
        cache.free_head_ = id;
        cache.num_free_ids_ += 1;

        return;
    }

    if (cache.num_overflow_ids_ < ids_per_cache_) {
        release_node.freeNode.subNext = cache.overflow_head_;
        cache.overflow_head_ = id;
        cache.num_overflow_ids_ += 1;
    }

    // If overflow cache is too big return it to the global free list
    if (cache.num_overflow_ids_ == ids_per_cache_) {
        FreeHead cur_head = free_head_.load(std::memory_order_relaxed);
        FreeHead new_head;
        new_head.head = cache.overflow_head_;
        Node &new_node = store_[cache.overflow_head_];

        do {
            new_head.gen = cur_head.gen + 1;
            new_node.freeNode.globalNext = cur_head.head;
        } while (!free_head_.compare_exchange_weak(
            cur_head, new_head, std::memory_order_release,
            std::memory_order_relaxed));

        cache.overflow_head_ = ~0u;
        cache.num_overflow_ids_ = 0;
    }
}

template <typename K, typename V, template <typename> typename StoreT>
void IDMap<K, V, StoreT>::bulkRelease(Cache &cache, K *keys,
                                      uint32_t num_keys)
{
    if (num_keys == 0) return;

    // The trick with this function is that the sublists added to the
    // global free list need to be exactly ids_per_cache_ in size
    // num_entities may not be divisible, so use the cache for any overflow

    auto linkToNext = [this, keys](int32_t idx) {
        uint32_t cur_idx = keys[idx].id;
        uint32_t next_idx = keys[idx + 1].id;

        Node &node = store_[cur_idx];
        node.gen.store(node.gen.load(std::memory_order_relaxed) + 1);
        node.freeNode.subNext = next_idx;
        node.freeNode.globalNext = 1;
    };
    
    int32_t base_idx;
    uint32_t num_remaining;
    Node *global_tail_node = nullptr;
    for (base_idx = 0; base_idx < (int)num_keys;
         base_idx += ids_per_cache_) {

        num_remaining = num_keys - base_idx;
        if (num_remaining < ids_per_cache_) {
            break;
        }

        uint32_t head_id = keys[base_idx].id;

        for (int sub_idx = 0; sub_idx < (int)ids_per_cache_; sub_idx++) {
            int idx = base_idx + sub_idx;
            linkToNext(idx);
        }

        Node &last_node = store_[keys[base_idx + ids_per_cache_ - 1].id];
        last_node.gen.store(last_node.gen.load(std::memory_order_relaxed) + 1);
        last_node.freeNode.subNext = ~0u;
        last_node.freeNode.globalNext = 1;

        if (global_tail_node != nullptr) {
            global_tail_node->freeNode.globalNext = head_id;
        }

        global_tail_node = &store_[head_id];
    }

    // The final chunk has an odd size we need to take care of
    if (num_remaining != ids_per_cache_) {
        uint32_t start_id = keys[base_idx].id;
        for (int idx = base_idx; idx < (int)num_keys - 1; idx++) {
            linkToNext(idx);
        }

        Node &tail_node = store_[keys[num_keys - 1].id];
        tail_node.gen.store(tail_node.gen.load(std::memory_order_relaxed) + 1);
        tail_node.freeNode.globalNext = 1;
        tail_node.freeNode.subNext = cache.overflow_head_;

        uint32_t num_from_overflow = ids_per_cache_ - num_remaining;
        if (cache.num_overflow_ids_ < num_from_overflow) {
            // The extra IDs fit in the overflow cache
            cache.overflow_head_ = start_id;
            cache.num_overflow_ids_ += num_remaining;
        } else {
            // The extra IDs don't fit in the overflow cache, need to add
            // to global list
            uint32_t next_id = cache.overflow_head_;
            Node *overflow_node;
            for (int i = 0; i < (int)num_from_overflow; i++) {
                overflow_node = &store_[next_id];
                next_id = overflow_node->freeNode.subNext;
            }

            overflow_node->freeNode.subNext = ~0u;
            cache.overflow_head_ = next_id;
            cache.num_overflow_ids_ -= num_from_overflow;

            if (global_tail_node != nullptr) {
                global_tail_node->freeNode.globalNext = start_id;
            }
            global_tail_node = &store_[start_id];
        }
    }

    // If global_tail_node is still unset, there is no full sublist to add
    // to the global list
    if (global_tail_node == nullptr) {
        return;
    }

    uint32_t new_global_head = keys[0].id;

    FreeHead cur_head = free_head_.load(std::memory_order_relaxed);
    FreeHead new_head;
    new_head.head = new_global_head;

    do {
        new_head.gen = cur_head.gen + 1;
        global_tail_node->freeNode.globalNext = cur_head.head;
    } while (!free_head_.compare_exchange_weak(
        cur_head, new_head, std::memory_order_release,
        std::memory_order_relaxed));
}

}
