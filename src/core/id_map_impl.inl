#pragma once

#include <madrona/cached_freelist.hpp>

namespace madrona {

template <typename T>
CachedFreeList<T>::Cache::Cache()
    : freeHead(~0u),
      numFreeIDs(0),
      overflowHead(~0u),
      numOverflowIDs(0)
{}

template <typename K, typename V, bool expandable>
IDMap<K, V, expandable>::IDMap(uint32_t init_capacity)
    : head_(FreeHead {
          .gen = 0,
          .head = ~0u,
      })
{}

template <typename K, typename V, bool expandable>
K IDMap<K, V, expandable>::acquireID(Cache &cache)
{
    auto assignCachedID = [this](uint32_t *head) {
        uint32_t new_id = *head;
        GenValue *free_loc = getGenLoc(new_id);

        // archetype is overloaded for GenLoc in the freelist
        // to represent multiple contiguous free IDs
        uint32_t num_contiguous = free_loc->loc.archetype;

        if (num_contiguous == 1) {
            *head = free_loc->loc.row;
        } else {
            uint32_t next_free = new_id + 1;
            GenLoc *next_free_loc = getGenLoc(next_free);
            *next_free_loc = GenLoc {
                .loc = Loc {
                    .archetype = num_contiguous - 1,
                    .row = free_loc->loc.row,
                },
                .gen = 0,
            };
            *head = next_free;
        }

        return Entity {
            .gen = free_loc->gen,
            .id = new_id,
        };
    };

    // First, check if there is a free entity in the overflow cache
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
    GenLoc *cur_head_loc;

    do {
        if (cur_head.head == ~0u) {
            break;
        }

        new_head.gen = cur_head.gen + 1;
        cur_head_loc = getGenLoc(cur_head.head);

        // On the global list, 'archetype' acts as the next pointer. This
        // preserves 'row' for the sublists added by the caches, each of
        // which is guaranteed to be ICfg::idsPerCache in size.
        // This works, because there are guaranteed to be no contiguous
        // blocks by the time a sublist is added to the global list, so
        // the value of archetype is no longer needed in the sublist
        // context
        new_head.head = cur_head_loc->loc.archetype;
    } while (!free_head_.compare_exchange_weak(
        cur_head, new_head, std::memory_order_release,
        std::memory_order_acquire));

    uint32_t free_ids = cur_head.head;

    if (free_ids != ~0u) {
        // Assign archetype to 1 (id block of size 1) so as to not confuse
        // the cache's processing of the freelist
        cur_head_loc->loc.archetype = 1;

        cache.free_head_ = free_ids;
        cache.num_free_ids_ = ICfg::idsPerCache - 1;
        return assignCachedID(&cache.free_head_);
    }

    // No free IDs at all, expand the ID store
    uint32_t block_start;
    {
        std::lock_guard lock(expand_lock_);

        // Note there is no double checked locking here.
        // It's possible that free ids have been returned at this point,
        // but if there is that much contention overallocating IDs seems
        // relatively harmless

        block_start = num_ids_;
        num_ids_ += ICfg::idsPerCache;
        store_.expand(num_ids_);
    }

    uint32_t first_id = block_start;

    GenLoc *entity_loc = getGenLoc(first_id);
    entity_loc->gen = 0;

    uint32_t free_start = block_start + 1;
    GenLoc *free_loc = getGenLoc(free_start);

    *free_loc = {
        .loc = Loc {
            // In the free list, archetype is overloaded
            // to handle contiguous free elements
            .archetype = ICfg::idsPerCache - 1,
            .row = ~0u,
        },
        .gen = 0,
    };

    cache.free_head_ = free_start;
    cache.num_free_ids_ = ICfg::idsPerCache - 1;

    return Entity {
        .gen = 0,
        .id = first_id,
    };

}

template <typename K, typename V, bool expandable>
void CachedFreelist<T>::releaseID(Cache &cache, uint32_t id)
{
    GenLoc *gen_loc = getGenLoc(e.id);
    gen_loc->gen++;
    gen_loc->loc.archetype = 1;

    if (cache.num_free_ids_ < ICfg::idsPerCache) {
        gen_loc->loc.row = cache.free_head_;
        cache.free_head_ = e.id;
        cache.num_free_ids_ += 1;

        return;
    }

    if (cache.num_overflow_ids_ < ICfg::idsPerCache) {
        gen_loc->loc.row = cache.overflow_head_;
        cache.overflow_head_ = e.id;
        cache.num_overflow_ids_ += 1;
    }

    // If overflow cache is too big return it to the global free list
    if (cache.num_overflow_ids_ == ICfg::idsPerCache) {
        FreeHead cur_head = free_head_.load(std::memory_order_relaxed);
        FreeHead new_head;
        new_head.head = cache.overflow_head_;
        GenLoc *new_loc = getGenLoc(cache.overflow_head_);

        do {
            new_head.gen = cur_head.gen + 1;
            new_loc->loc.archetype = cur_head.head;
        } while (free_head_.compare_exchange_weak(
            cur_head, new_head, std::memory_order_release,
            std::memory_order_relaxed));

        cache.overflow_head_ = ~0u;
        cache.num_overflow_ids_ = 0;
    }
}

template <typename K, typename V, bool expandable>
void IDMap::bulkRelease(Cache &cache, K *ids,
                        uint32_t num_ids)
{
    if (num_entities == 0) return;

    // The trick with this function is that the sublists added to the
    // global free list need to be exactly ICfg::idsPerCache in size
    // num_entities may not be divisible, so use the cache for any overflow

    auto linkToNext = [this, entities](int idx) {
        Entity cur = entities[idx];
        Entity next = entities[idx + 1];

        GenLoc *gen_loc = getGenLoc(cur.id);
        gen_loc->gen++;
        gen_loc->loc.row = next.id;
        gen_loc->loc.archetype = 1;
    };
    
    int base_idx;
    uint32_t num_remaining;
    GenLoc *global_tail_loc = nullptr;
    for (base_idx = 0; base_idx < (int)num_entities;
         base_idx += ICfg::idsPerCache) {

        num_remaining = num_entities - base_idx;
        if (num_remaining < ICfg::idsPerCache) {
            break;
        }

        uint32_t head_id = entities[base_idx].id;

        for (int sub_idx = 0; sub_idx < (int)ICfg::idsPerCache; sub_idx++) {
            int idx = base_idx + sub_idx;
            linkToNext(idx);
        }

        GenLoc *last_loc = getGenLoc(entities[ICfg::idsPerCache - 1].id);
        last_loc->gen++;
        last_loc->loc.row = ~0u;
        last_loc->loc.archetype = 1;

        if (global_tail_loc != nullptr) {
            global_tail_loc->loc.archetype = head_id;
        }

        global_tail_loc = getGenLoc(head_id);
    }

    // The final chunk has an odd size we need to take care of
    if (num_remaining != ICfg::idsPerCache) {
        uint32_t start_id = entities[base_idx].id;
        for (int idx = base_idx; idx < (int)num_entities - 1; idx++) {
            linkToNext(idx);
        }

        GenLoc *tail_loc = getGenLoc(entities[num_entities - 1].id);
        tail_loc->gen++;
        tail_loc->loc.archetype = 1;
        tail_loc->loc.row = cache.overflow_head_;

        uint32_t num_from_overflow = ICfg::idsPerCache - num_remaining;
        if (cache.num_overflow_ids_ < num_from_overflow) {
            // The extra IDs fit in the overflow cache
            cache.overflow_head_ = start_id;
            cache.num_overflow_ids_ += num_remaining;
        } else {
            // The extra IDs don't fit in the overflow cache, need to add to global list
            uint32_t next_id = cache.overflow_head_;
            GenLoc *overflow_loc;
            for (int i = 0; i < (int)num_from_overflow; i++) {
                overflow_loc = getGenLoc(next_id);
                next_id = overflow_loc->loc.row;
            }

            overflow_loc->loc.row = ~0u;
            cache.overflow_head_ = next_id;
            cache.num_overflow_ids_ -= num_from_overflow;

            if (global_tail_loc != nullptr) {
                global_tail_loc->loc.archetype = start_id;
            }
            global_tail_loc = getGenLoc(start_id);
        }
    }

    // If global_tail_loc is still unset, there is no full sublist to add
    // to the global list
    if (global_tail_loc != nullptr) {
        return;
    }

    uint32_t new_global_head = entities[0].id;

    FreeHead cur_head = free_head_.load(std::memory_order_relaxed);
    FreeHead new_head;
    new_head.head = new_global_head;

    do {
        new_head.gen = cur_head.gen + 1;
        global_tail_loc->loc.archetype = cur_head.head;
    } while (free_head_.compare_exchange_weak(
        cur_head, new_head, std::memory_order_release,
        std::memory_order_relaxed));
}

}
