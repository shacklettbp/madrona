#pragma once

#include <cstdint>
#include <type_traits>
#include <madrona/crash.hpp>

namespace madrona {
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
    return (offset + alignment - 1) & -alignment;
}

constexpr inline bool isPower2(uint64_t v)
{
    return (v & (v - 1)) == 0;
}

constexpr inline uint32_t intLog2(uint32_t v)
{
    return sizeof(unsigned int) * 8 - __builtin_clz(v) - 1;
}

}
}
