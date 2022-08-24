#include <madrona/virtual.hpp>
#include <madrona/crash.hpp>
#include <madrona/utils.hpp>

#if defined(__linux__) or defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace madrona {

VirtualRegion::VirtualRegion(uint64_t max_chunks, uint64_t init_chunks)
{
#if defined(__linux__) or defined(__APPLE__)
    uint64_t overalign_size =
        (max_chunks << chunkShift()) + chunkSize() - 1;

#ifdef __linux__
    constexpr int mmap_init_flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#elif defined(__APPLE__)
    constexpr int mmap_init_flags = MAP_PRIVATE | MAP_ANON;
#endif

    void *base_attempt =
        mmap(nullptr, overalign_size, PROT_NONE, mmap_init_flags, -1, 0);

    if (base_attempt == MAP_FAILED) [[unlikely]] {
        FATAL("Failed to allocate %lu bytes of virtual address space\n",
              overalign_size);
    }

    uint64_t offset = (uintptr_t)base_attempt & (chunkSize() - 1);
    uint64_t extra_amount = chunkSize() - offset;

    if (extra_amount > 0) {
        munmap(base_attempt, extra_amount);
    }

    char *base = (char *)base_attempt + extra_amount;
    uint64_t total_size = overalign_size - extra_amount;

#elif _WIN32
    STATIC_UNIMPLEMENTED();
#else
    STATIC_UNIMPLEMENTED();
#endif

    base_ = base;
    total_size_ = total_size;

    if (init_chunks > 0) {
        commit(0, init_chunks);
    }
}

VirtualRegion::~VirtualRegion()
{
#if defined(__linux__) or defined(__APPLE__)
    munmap(base_, total_size_);
#elif defined(_WIN32)
    STATIC_UNIMPLEMENTED();
#else
    STATIC_UNIMPLEMENTED();
#endif
}

void VirtualRegion::commit(uint64_t start_chunk, uint64_t num_chunks)
{
#if defined(__linux__) or defined(__APPLE__)
    void *res = mmap(base_ + (start_chunk << chunkShift()),
        num_chunks << chunkShift(), PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);

    if (res == MAP_FAILED) {
        FATAL("Failed to commit %lu chunks for VirtualRegion", num_chunks);
    }
#elif defined(_WIN32)
    STATIC_UNIMPLEMENTED();
#else
    STATIC_UNIMPLEMENTED();
#endif
}

void VirtualRegion::decommit(uint64_t start_chunk, uint64_t num_chunks)
{
    int res = madvise(base_ + (start_chunk << chunkShift()),
                      num_chunks << chunkShift(), MADV_FREE);

    if (res != 0) {
        FATAL("Failed to decommit %lu chunks for VirtualRegion", num_chunks);
    }
}

VirtualStore::VirtualStore(uint32_t bytes_per_item, uint32_t max_items)
    : region_(utils::divideRoundUp((uint64_t)bytes_per_item * max_items,
        VirtualRegion::chunkSize()), 0),
      bytes_per_item_(bytes_per_item),
      committed_chunks_(0),
      committed_items_(0)
{}

void VirtualStore::expand(uint32_t num_items)
{
    if (num_items > committed_items_) {
        region_.commit(committed_chunks_, 1);
        committed_chunks_++;

        committed_items_ = committed_chunks_ * VirtualRegion::chunkSize() /
            bytes_per_item_;
    }
}

void VirtualStore::shrink(uint32_t num_items)
{
    uint32_t num_excess = committed_items_ - num_items;

    if (num_excess * bytes_per_item_ > VirtualRegion::chunkSize()) {
        committed_chunks_--;
        region_.decommit(committed_chunks_, 1);
        committed_items_ = committed_chunks_ * VirtualRegion::chunkSize() /
            bytes_per_item_;
    }
}

}
