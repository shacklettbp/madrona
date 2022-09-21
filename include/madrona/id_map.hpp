#pragma once

namespace madrona {

template <typename K, typename V, typename StoreT>
class IDMap {
public:
    struct Cache {
        Cache();
        Cache(const Cache &) = delete;

        uint32_t freeHead;
        uint32_t numFreeIDs;
        uint32_t overflowHead;
        uint32_t numOverflowIDs;
    };

    IDMap(uint32_t init_capacity);

    K acquireID(Cache &cache);

    void releaseID(Cache &cache, K id);
    void bulkRelease(Cache &cache, K *ids, uint32_t num_ids);

private:
    struct Node {
        union {
            T val;
            struct {
                uint32_t subNext;
                uint32_t globalNext;
                uint32_t gen;
            };
        }
    };
    static_assert(sizeof(Node) <= sizeof(T));
    static_assert(offsetof(Node, subNext) == 0);

    struct alignas(std::atomic_uint64_t) FreeHead {
        uint32_t gen;
        uint32_t head;
    };

    template <bool expandable> struct Store;

    inline V & getValue(uint32_t idx)
    {
        return ((V *)store_.data())[idx];
    }

    inline const V & getVal(uint32_t idx) const
    {
        return ((const V *)store_.data())[idx];
    }

    std::atomic<FreeHead> head_;
    StoreT_ store_;
};

}
