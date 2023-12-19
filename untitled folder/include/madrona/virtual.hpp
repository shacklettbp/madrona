/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

#include <madrona/types.hpp>
#include <madrona/utils.hpp>

namespace madrona {

class VirtualRegion {
public:
    VirtualRegion(uint64_t max_bytes, uint64_t chunk_shift,
                  uint64_t alignment, uint64_t init_chunks = 0);
    VirtualRegion(const VirtualRegion &) = delete;
    VirtualRegion(VirtualRegion &&o);

    ~VirtualRegion();


    inline void *ptr() const { return aligned_; }
    void commitChunks(uint64_t start_chunk, uint64_t num_chunks);
    void decommitChunks(uint64_t start_chunk, uint64_t num_chunks);

    inline uint64_t chunkSize() const { return 1_u64 << chunk_shift_; }

private:
    struct Init;
    inline VirtualRegion(Init init);

    char * base_;
    char * aligned_;
    uint64_t chunk_shift_;
    uint64_t total_size_;
};

class VirtualStore {
public:
    VirtualStore(uint32_t bytes_per_item,
                 uint32_t item_alignment,
                 uint32_t start_offset,
                 uint32_t max_items);

    inline void * operator[](uint32_t idx) 
    {
        return (char *)data_ + bytes_per_item_ * idx;
    }

    inline const void * operator[](uint32_t idx) const
    {
        return (char *)data_ + bytes_per_item_ * idx;
    }

    void expand(uint32_t num_items);
    void shrink(uint32_t num_items);

    inline void * data() const { return data_; }

    inline uint32_t numBytesPerItem() const { return bytes_per_item_; }

private:
    VirtualRegion region_;
    void *const data_;
    uint32_t bytes_per_item_;
    uint32_t start_offset_;
    uint32_t committed_chunks_;
    uint32_t committed_items_;
};

template <typename T>
class VirtualArray {
public:
    VirtualArray(uint32_t start_offset, uint32_t max_items)
        : store_(sizeof(T), alignof(T), start_offset, max_items),
          size_(0)
    {}

    T & push_back(T &&v)
    {
        uint32_t idx = size_++;

        store_.expand(size_);
        return *new (&operator[](idx)) T(v);
    }

    T & push_back(const T &&v)
    {
        uint32_t idx = size_++;

        store_.expand(size_);
        return *new (&operator[](idx)) T(v);
    }

    template <typename Fn>
    void resize(uint32_t new_size, Fn &&fn)
    {
        if (new_size > size_) {
            store_.expand(new_size);

            for (uint32_t i = size_; i < new_size; i++) {
                fn(&operator[](i));
            }
        } else if (new_size < size_) {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                for (size_t i = new_size; i < size_; i++) {
                    operator[](i).~T();
                }
            }

            store_.shrink(new_size);
        }

        size_ = new_size;
    } 

    inline T & operator[](uint32_t idx)
    {
        return ((T *)store_.data())[idx];
    }

    inline const T & operator[](uint32_t idx) const
    {
        return ((T *)store_.data())[idx];
    }

    uint32_t size() const { return size_; }

private:
    VirtualStore store_;
    uint32_t size_;
};

}
