#pragma once

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
        uint32_t free_head_;
        uint32_t num_free_ids_;
        uint32_t overflow_head_;
        uint32_t num_overflow_ids_;

    friend class IDMap;
    };

    struct FreeNode {
        uint32_t subNext;
        uint32_t globalNext;
    };

    struct Node {
        union {
            V val;
            FreeNode freeNode;
        };
        uint32_t gen;
    };

    IDMap(uint32_t init_capacity);

    K acquireID(Cache &cache);

    void releaseID(Cache &cache, uint32_t id);
    inline void releaseID(Cache &cache, K k)
    {
        releaseID(cache, k.id);
    }

    void bulkRelease(Cache &cache, K *keys, uint32_t num_keys);

    inline V lookup(K k) const
    {
        const Node &node = store_[k.id];

        if (node.gen != k.gen) {
            return V::none();
        }

        return node.val;
    }

    inline bool present(K k) const
    {
        const Node &node = store_[k.id];
        return node.gen == k.gen;
    }

    inline V & getRef(K k)
    {
        Node &node = store_[k.id];
        assert(node.gen == k.gen);

        return node.val;
    }

    inline V & getRef(uint32_t id)
    {
        return store_[id].val;
    }

private:
    using Store = StoreT<Node>;
   
    static_assert(sizeof(FreeNode) <= sizeof(V));

    struct alignas(std::atomic_uint64_t) FreeHead {
        uint32_t gen;
        uint32_t head;
    };

    std::atomic<FreeHead> free_head_;
    [[no_unique_address]] Store store_;

    static_assert(decltype(free_head_)::is_always_lock_free);

    static constexpr uint32_t ids_per_cache_ = 64;

    friend Store;
};

}
