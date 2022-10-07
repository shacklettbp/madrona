#pragma once

#include <madrona/memory.hpp>

namespace madrona {

// A lock-free list implemented as a linked-list of blocks.
// Iterable from front to back but not indexable
template <typename T>
class BlockList {
    struct Block;
public:
    using RefT = std::add_lvalue_reference_t<T>;

    BlockList();
    BlockList(const BlockList &) = delete;
    BlockList(BlockList &&o);
    ~BlockList();

    inline RefT push_back(T &&v)
    {

    }

    class Iter {
        Block *block_;
        IdxT offset_;

        inline Iter & operator++();
        inline Iter operator++(int);

        friend bool operator==(const Iter &a, const Iter &b)
        {
            return a.block_ == b.block_ && a.offset_ == b.offset_;
        }

        friend bool operator!=(const Iter &a, const Iter &b)
        {
            return !(a == b);
        }
    };

    inline Iter begin()
    {
        return Iter {
            .block = head_block_,
            .offset_ = 0,
        };
    }

    inline Iter end()
    {
        return Iter {
            .block = tail_block_,
            .cur_offset_ = tail_block_ ? tail_block_->metadata.numItems : 0,
        };
    }

private:
    struct BlockMetadata {
        Block *next;
        uint32_t numItems;
    };
    static constexpr uint64_t items_per_block_ =
        (OSAlloc::chunkSize() - sizeof(BlockMetadata)) / sizeof(T);
    struct Block {
        T data[items_per_block_];
        BlockMetadata metadata;
    };

    static_assert(sizeof(Block) <= OSAlloc::chunkSize());
    Block *head_block_;
    Block *tail_block_;

friend class Iter;
};

template <typename T>
auto BlockList<T>::Iter::operator++() -> Iter &
{
    ++offset_;

    if (offset_ == items_per_block_) {
        offset_ = 0;
        block_ = block_->next;
    }
}

template <typename T>
auto BlockList<T>::Iter::operator++(int) -> Iter
{
    Iter next = *this;
    return ++next;
}

}
