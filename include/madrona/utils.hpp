/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <madrona/crash.hpp>

namespace madrona {

namespace utils {

#ifdef MADRONA_GPU_MODE

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

#endif

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
    return (offset + alignment - 1) & -alignment;
}

inline void *alignPtr(void *ptr, uintptr_t alignment)
{
    uintptr_t base = (uintptr_t)ptr;
    uintptr_t aligned = roundUpPow2(base, alignment);
    uintptr_t offset = aligned - base;
    return (char *)ptr + offset;
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
    return v == 1 ? 1 : (1u << (32u - __builtin_clz(v - 1)));
}

constexpr inline uint64_t int64NextPow2(uint64_t v)
{
    int clz;
    if constexpr (std::is_same_v<uint64_t, unsigned long>) {
        clz = __builtin_clzl(v - 1);
    } else if constexpr (std::is_same_v<uint64_t, unsigned long long>) {
        clz = __builtin_clzll(v - 1);
    }

    return v == 1 ? 1 : (1u << (64u - clz));
}

constexpr inline uint32_t int32Log2(uint32_t v)
{
    return sizeof(unsigned int) * 8 - __builtin_clz(v) - 1;
}

constexpr inline uint64_t int64Log2(uint64_t v)
{
    return sizeof(unsigned long long) * 8 - __builtin_clzll(v) - 1;
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

template <typename> struct PackDelegator;
template <template <typename...> typename T, typename ...Args>
struct PackDelegator<T<Args...>> {
    template <typename Fn>
    static auto call(Fn &&fn) -> decltype(fn.template operator()<Args...>())
    {
        return fn.template operator()<Args...>();
    }
};

}
}
