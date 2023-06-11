namespace madrona {

StackAlloc::Frame StackAlloc::push()
{
    return Frame {
        cur_chunk_ + chunk_offset_,
    };
}

void * StackAlloc::alloc(CountT num_bytes, CountT alignment)
{
    CountT alloc_offset =
        (CountT)utils::roundUpPow2(chunk_offset_, (uint32_t)alignment);
    CountT new_offset = alloc_offset + num_bytes;

    if (new_offset <= chunk_size_) {
        void *start = cur_chunk_ + alloc_offset;
        chunk_offset_ = new_offset;

        return start;
    }

    alloc_offset =
        (CountT)utils::roundUpPow2(sizeof(ChunkMetadata), (uint32_t)alignment);
    new_offset = alloc_offset + num_bytes;

    if (new_offset > chunk_size_) [[unlikely]] {
        allocTooLarge();
    }

    char *new_chunk = StackAlloc::newChunk(chunk_size_);

    auto *cur_metadata = (ChunkMetadata *)cur_chunk_;
    cur_metadata->next = (ChunkMetadata *)new_chunk;

    cur_chunk_ = new_chunk;
    chunk_offset_ = new_offset;

    return new_chunk + new_offset;
}

template <typename T>
T * StackAlloc::alloc()
{
    return (T *)alloc(sizeof(T), alignof(T));
}

template <typename T>
T * StackAlloc::allocN(CountT num_elems)
{
    return (T *)alloc(sizeof(T) * num_elems, alignof(T));
}

}
