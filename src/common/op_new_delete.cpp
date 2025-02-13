#include <madrona/memory.hpp>

#include <cstdlib>
#include <cstdio>

namespace madrona {

namespace {

inline void * opNewImpl(size_t num_bytes) noexcept
{
    void *ptr = rawAlloc(num_bytes);
    if (!ptr) [[unlikely]] {
        fprintf(stderr, "OOM: %zu\n", num_bytes);
        fflush(stderr);
        std::abort();
    }

    return ptr;
}

inline void opDeleteImpl(void *ptr) noexcept
{
    rawDealloc(ptr);
}

inline void * opNewAlignImpl(size_t num_bytes, std::align_val_t al) noexcept
{
    size_t alignment = static_cast<size_t>(al);
    if (alignment < sizeof(void *)) {
        alignment = sizeof(void *);
    }

    static_assert(sizeof(al) == sizeof(size_t));
    void *ptr = rawAllocAligned(num_bytes, alignment);
    if (!ptr) [[unlikely]] {
        fprintf(stderr, "OOM: %zu %zu\n", num_bytes, (size_t)al);
        fflush(stderr);
        std::abort();
    }

    return ptr;
}

inline void opDeleteAlignImpl(void *ptr, std::align_val_t al) noexcept
{
    (void)al;
    rawDeallocAligned(ptr);
}

}

}

// madrona-libcxx is compiled without operator new and delete,
// because libc++'s static hermetic mode marks operator new and delete
// as hidden symbols. Unfortunately, this breaks ASAN's (and our own) ability
// to export operator new and operator delete outside of the shared library
// executable. Therefore we disable operator new and delete in libcxx and
// libcxxabi and must provide them here.

// Unaligned versions

#ifdef MADRONA_WINDOWS
#define MADRONA_NEWDEL_VIS
#else
#define MADRONA_NEWDEL_VIS MADRONA_EXPORT
#endif

MADRONA_NEWDEL_VIS void * operator new(size_t num_bytes)
{
    return ::madrona::opNewImpl(num_bytes);
}

MADRONA_NEWDEL_VIS void operator delete(void *ptr) noexcept
{
    ::madrona::opDeleteImpl(ptr);
}

MADRONA_NEWDEL_VIS void * operator new(
    size_t num_bytes, const std::nothrow_t &) noexcept
{
    return ::madrona::opNewImpl(num_bytes);
}

MADRONA_NEWDEL_VIS void operator delete(
    void *ptr, const std::nothrow_t &) noexcept
{
    ::madrona::opDeleteImpl(ptr);
}

MADRONA_NEWDEL_VIS void * operator new[](size_t num_bytes)
{
    return ::madrona::opNewImpl(num_bytes);
}

MADRONA_NEWDEL_VIS void operator delete[](void *ptr) noexcept
{
    ::madrona::opDeleteImpl(ptr);
}

MADRONA_NEWDEL_VIS void * operator new[](
    size_t num_bytes, const std::nothrow_t &) noexcept
{
    return ::madrona::opNewImpl(num_bytes);
}

MADRONA_NEWDEL_VIS void operator delete[](
    void *ptr, const std::nothrow_t &) noexcept
{
    ::madrona::opDeleteImpl(ptr);
}

// Aligned versions

MADRONA_NEWDEL_VIS void * operator new(size_t num_bytes, std::align_val_t al)
{
    return ::madrona::opNewAlignImpl(num_bytes, al);
}

MADRONA_NEWDEL_VIS void operator delete(void *ptr, std::align_val_t al) noexcept
{
    ::madrona::opDeleteAlignImpl(ptr, al);
}

MADRONA_NEWDEL_VIS void * operator new(
    size_t num_bytes, std::align_val_t al, const std::nothrow_t &) noexcept
{
    return ::madrona::opNewAlignImpl(num_bytes, al);
}

MADRONA_NEWDEL_VIS void operator delete(
    void *ptr, std::align_val_t al, const std::nothrow_t &) noexcept
{
    ::madrona::opDeleteAlignImpl(ptr, al);
}

MADRONA_NEWDEL_VIS void * operator new[](size_t num_bytes, std::align_val_t al)
{
    return ::madrona::opNewAlignImpl(num_bytes, al);
}

MADRONA_NEWDEL_VIS void operator delete[](void *ptr, std::align_val_t al) noexcept
{
    ::madrona::opDeleteAlignImpl(ptr, al);
}

MADRONA_NEWDEL_VIS void * operator new[](
    size_t num_bytes, std::align_val_t al, const std::nothrow_t &) noexcept
{
    return ::madrona::opNewAlignImpl(num_bytes, al);
}

MADRONA_NEWDEL_VIS void operator delete[](
    void *ptr, std::align_val_t al, const std::nothrow_t &) noexcept
{
    ::madrona::opDeleteAlignImpl(ptr, al);
}
