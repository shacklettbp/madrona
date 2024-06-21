/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#include <cassert>
#include <madrona/utils.hpp>

namespace madrona {

void * rawAllocAligned(size_t num_bytes, size_t alignment)
{
#if defined(_LIBCPP_VERSION)
    return std::aligned_alloc(alignment, num_bytes);
#elif defined(MADRONA_MSVC)
    return _aligned_malloc(num_bytes, alignment);
#else
    STATIC_UNIMPLEMENTED();
#endif
}

void rawDeallocAligned(void *ptr)
{
#if defined(_LIBCPP_VERSION)
    free(ptr);
#elif defined(MADRONA_MSVC)
    _aligned_free(ptr);
#else
    STATIC_UNIMPLEMENTED();
#endif
}

PolyAlloc::PolyAlloc(void *state,
                     void *(*alloc_ptr)(void *, size_t),
                     void (*dealloc_ptr)(void *, void *))
    : state_(state),
      alloc_ptr_(alloc_ptr),
      dealloc_ptr_(dealloc_ptr)
{}

void *PolyAlloc::alloc(size_t num_bytes)
{
    return alloc_ptr_(state_, num_bytes);
}

void PolyAlloc::dealloc(void *ptr)
{
    dealloc_ptr_(state_, ptr);
}

template <typename A>
PolyAlloc Allocator<A>::getPoly()
{
    return PolyAlloc(this, &allocStatic, &deallocStatic);
}

template <typename A>
void * Allocator<A>::allocStatic(void *state, size_t num_bytes)
{
    return static_cast<A *>(state)->alloc(num_bytes);
}

template <typename A>
void Allocator<A>::deallocStatic(void *state, void *ptr)
{
    static_cast<A *>(state)->dealloc(ptr);
}

void * AllocContext::alloc(size_t num_bytes)
{
    return cur_alloc_.alloc(num_bytes);
}

void AllocContext::dealloc(void *ptr)
{
    cur_alloc_.dealloc(ptr);
}

template <typename T, typename ...Args>
T * AllocContext::make(Args &&...args)
{
    auto ptr = (T *)alloc(sizeof(T));
    new (ptr) T(std::forward(args)...);

    return ptr;
}

template <typename T>
void AllocContext::destroy(T *ptr)
{
    ptr->~T();
    dealloc(ptr);
}

template <typename A>
inline AllocScope AllocContext::scope(A &alloc)
{
    return AllocScope(alloc.getPoly(), cur_scope_, this);
}

template <typename A, typename Fn, typename ...Args>
auto AllocContext::with(A &alloc, Fn &&fn, Args &&...args) ->
    decltype(Fn(args...))
{
    AllocScope tracker = scope(alloc);
    return fn(std::forward(args)...);
}

void * DefaultAlloc::alloc(size_t num_bytes)
{
    return rawAllocAligned(utils::roundUpPow2(num_bytes, MADRONA_CACHE_LINE),
        MADRONA_CACHE_LINE);
}

void DefaultAlloc::dealloc(void *ptr)
{
    rawDeallocAligned(ptr);
}

}
