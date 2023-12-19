/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/macros.hpp>
#include <madrona/sync.hpp>
#include <madrona/types.hpp>

#include <atomic>
#include <cstdint>

namespace madrona {

template <typename K, typename V, template <typename> typename StoreT>
class IDMap {
public:
    class Cache {
    public:
        Cache();
        Cache(const Cache &) = delete;

    private:
        int32_t free_head_;
        int32_t num_free_ids_;
        int32_t overflow_head_;
        int32_t num_overflow_ids_;

    friend class IDMap;
    };

    struct FreeNode {
        int32_t subNext;
        int32_t globalNext;
    };

    struct Node {
        union {
            V val;
            FreeNode freeNode;
        };
        AtomicU32 gen;
    };

    IDMap(CountT init_capacity);

    inline K acquireID(Cache &cache);

    inline void releaseID(Cache &cache, int32_t id);
    inline void releaseID(Cache &cache, K k)
    {
        releaseID(cache, k.id);
    }

    inline void bulkRelease(Cache &cache, K *keys, CountT num_keys);

    inline V lookup(K k) const
    {
        const Node &node = store_[k.id];

        if (node.gen.load_relaxed() != k.gen) {
            return V::none();
        }

        return node.val;
    }

    inline bool present(K k) const
    {
        const Node &node = store_[k.id];
        return node.gen.load_relaxed() == k.gen;
    }

    inline V & getRef(K k)
    {
        Node &node = store_[k.id];
        assert(node.gen.load_relaxed() == k.gen);

        return node.val;
    }

    inline const V & getRef(K k) const
    {
        const Node &node = store_[k.id];
        assert(node.gen.load_relaxed() == k.gen);

        return node.val;
    }

    inline V & getRef(int32_t id)
    {
        return store_[id].val;
    }

    inline const V & getRef(int32_t id) const
    {
        return store_[id].val;
    }

#ifdef TSAN_ENABLED
    // These helpers are needed when TSAN is enabled to allow explicitly
    // marking Node::gen as acquired and released, which is currently done
    // with atomic_thread_fence by outside code after updating / reading
    // multiple IDs.

    inline void acquireGen(int32_t id)
    {
        TSAN_ACQUIRE(&store_[id].gen);
    }

    inline void releaseGen(int32_t id)
    {
        TSAN_RELEASE(&store_[id].gen);
    }
#endif

private:
    using Store = StoreT<Node>;
   
    static_assert(sizeof(FreeNode) <= sizeof(V));

    struct alignas(AtomicU64) FreeHead {
        uint32_t gen;
        int32_t head;
    };

    [[no_unique_address]] Store store_;
    alignas(MADRONA_CACHE_LINE) Atomic<FreeHead> free_head_;

    static constexpr CountT ids_per_cache_ = 64;
    static constexpr int32_t sentinel_ = 0xFFFF'FFFF_i32;

    friend Store;
};

}
