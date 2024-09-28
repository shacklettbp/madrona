/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/crash.hpp>
#include <madrona/macros.hpp>
#include <madrona/span.hpp>
#include <madrona/template_helpers.hpp>

#ifdef MADRONA_MSVC
#include <bit>
#endif
#include <cstring>
#include <cstdint>
#include <type_traits>

// FIXME: Ultimately it would be better to just change all uses of these
// functions to std::bit_width
#if defined(MADRONA_GPU_MODE)
inline int __builtin_clz(int v)
{
    return __clz(v);
}

inline int __builtin_clzl(long int v)
{
    return __clzll(v);
}

inline int __builtin_clzll(long long int v)
{
    return __clzll(v);
}

#elif defined(MADRONA_MSVC)

MADRONA_ALWAYS_INLINE constexpr inline int __builtin_clz(int v)
{
    using U = std::make_unsigned_t<int>;
    return sizeof(U) * 8 - std::bit_width(U(v));
}

MADRONA_ALWAYS_INLINE constexpr inline int __builtin_clzl(long int v)
{
    using U = std::make_unsigned_t<long int>;
    return sizeof(U) * 8 - std::bit_width(U(v));
}

MADRONA_ALWAYS_INLINE constexpr inline int __builtin_clzll(long long int v)
{
    using U = std::make_unsigned_t<long long int>;
    return sizeof(U) * 8 - std::bit_width(U(v));
}

#endif

namespace madrona {

template <typename T>
class ArrayQueue {
public:
    ArrayQueue(T *data, uint32_t capacity);

    void add(T t);
    T remove();

    uint32_t capacity() const;
    bool isEmpty() const;
    void clear();

private:
    uint32_t increment(uint32_t i);

    T *data_;
    uint32_t capacity_;
    uint32_t head_;
    uint32_t tail_;
};

namespace utils {

template <typename T>
constexpr inline T divideRoundUp(T a, T b)
{
    static_assert(std::is_integral_v<T>);

    return (a + (b - 1)) / b;
}

template <typename T>
constexpr inline T roundUp(T offset, T alignment)
{
    return divideRoundUp(offset, alignment) * alignment;
}

// alignment must be power of 2
constexpr inline uint64_t roundUpPow2(uint64_t offset, uint64_t alignment)
{
#ifdef MADRONA_MSVC
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
    return (offset + alignment - 1) & -alignment;
#ifdef MADRONA_MSVC
#pragma warning(pop)
#endif
}

inline uintptr_t alignPtrOffset(void *ptr, uintptr_t alignment)
{
    uintptr_t base = (uintptr_t)ptr;
    uintptr_t aligned = roundUpPow2(base, alignment);
    return aligned - base;
}

inline void *alignPtr(void *ptr, uintptr_t alignment)
{
    return (char *)ptr + alignPtrOffset(ptr, alignment);
}

constexpr inline bool isPower2(uint64_t v)
{
    return (v & (v - 1)) == 0;
}

constexpr inline bool isPower2(uint32_t v)
{
    return (v & (v - 1)) == 0;
}

constexpr inline uint32_t int32NextPow2(uint32_t v)
{
    return v == 1 ? 1 : (1u << (32u - __builtin_clz((int32_t)v - 1)));
}

constexpr inline uint64_t int64NextPow2(uint64_t v)
{
#if __cplusplus >= 202002L
    int clz;
#else
    int clz = 0;
#endif
    if constexpr (std::is_same_v<int64_t, long>) {
        clz = __builtin_clzl((long)v - 1);
    } else if constexpr (std::is_same_v<int64_t, long long>) {
        clz = __builtin_clzll((long long)v - 1);
    }

    return v == 1 ? 1 : (1u << (64u - clz));
}

constexpr inline uint32_t int32Log2(uint32_t v)
{
    static_assert(std::is_same_v<int32_t, int>);

    return sizeof(unsigned int) * 8 - __builtin_clz((int)v) - 1;
}

constexpr inline uint64_t int64Log2(uint64_t v)
{
    return sizeof(unsigned long long) * 8 - __builtin_clzll((long long)v) - 1;
}

// https://github.com/skeeto/hash-prospector
constexpr inline uint32_t int32Hash(uint32_t x)
{
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

inline int64_t computeBufferOffsets(const Span<const int64_t> chunk_sizes,
                                    Span<int64_t> out_offsets,
                                    int64_t pow2_alignment)
{
    int64_t num_total_bytes = chunk_sizes[0];

    for (int64_t i = 1; i < chunk_sizes.size(); i++) {
        int64_t cur_offset = roundUpPow2(num_total_bytes, pow2_alignment);
        out_offsets[i - 1] = cur_offset;

        num_total_bytes = cur_offset + chunk_sizes[i];
    }

    return roundUpPow2(num_total_bytes, pow2_alignment);
}

template <typename T>
inline void copyN(std::type_identity_t<T> *dst,
                  const std::type_identity_t<T> *src,
                  CountT num_elems);

template <typename T>
inline void zeroN(std::type_identity_t<T> *ptr, CountT num_elems);

template <typename T>
inline void fillN(std::type_identity_t<T> *ptr, T v, CountT num_elems);

constexpr inline uint32_t u32mulhi(uint32_t a, uint32_t b);

}

}

#include "utils.inl"
