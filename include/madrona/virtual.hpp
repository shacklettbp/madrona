#pragma once

#include <cstdint>

namespace madrona {

class VirtualRegion {
public:
    VirtualRegion(uint64_t max_bytes, uint64_t chunk_shift,
                  uint64_t aligment, uint64_t init_chunks);
    ~VirtualRegion();

    VirtualRegion(const VirtualRegion &) = delete;

    inline void *ptr() const { return base_; }
    void commit(uint64_t start_chunk, uint64_t num_chunks);
    void decommit(uint64_t start_chunk, uint64_t num_chunks);

    inline uint64_t chunkSize() const { return 1 << chunk_shift_; }

private:
    char *base_;
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
    void *data_;
    uint32_t bytes_per_item_;
    uint32_t start_offset_;
    uint32_t committed_chunks_;
    uint32_t committed_items_;
};

}
