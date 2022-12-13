#pragma once

#include <madrona/types.hpp>
#include <madrona/utils.hpp>

namespace madrona {

template <typename T,
          CountT desired_elems_per_block = 0>
class BlockList {
    struct BlockBase {};

    struct Metadata {
        BlockBase *next;
        CountT numElems;
    };

    template <CountT num_elems>
    struct Block : BlockBase {
        T arr[num_elems];
        Metadata metadata;
    };

    static constexpr inline CountT computeElemsPerBlock()
    {
        static_assert(desired_elems_per_block >= 0);

        if constexpr (desired_elems_per_block != 0) {
            return desired_elems_per_block;
        }

        constexpr CountT default_block_size = 1024;

        sizeof(T) 

        constexpr CountT num_elems = (default_block_size - sizeof(Metadata)) / sizeof(T);

        using TestT = Block<T>

        if constexpr (num_elems == 0) {
            return 1;
        }

        return num_elems;
    }

    static constexpr inline CountT per_block_ = computeElemsPerBlock();

    struct Block {
        T arr[per_block_];
        Metadata metadata;
    };


public:
    BlockList()
        : head_(nullptr)
    {}

    class Iter {
    public:

    private:
        Block *cur_block_;
        CountT cur_offset_;

    friend class BlockList;
    };

    Iter begin()
    {
    }

    Iter end()
    {
        return Iter {
        };
    }

private:
    static inline constexpr CountT num_elems_ =
        block_size / 

    union Block {
        Block *next;
        CountT numElems;
    };

    Block *head_;
};

}
