#pragma once

#include <cstdint>
#include <type_traits>

namespace madrona {

using u64 = uint64_t;
using i64 = int64_t;
using u32 = uint32_t;
using i32 = int32_t;
using u16 = uint16_t;
using i16 = int16_t;
using u8 = uint8_t;
using i8 = int8_t;
using f32 = float;

inline constexpr u32 operator "" _u32(unsigned long long v) 
{ 
    return uint32_t(v);
}

inline constexpr u64 operator "" _u64(unsigned long long v) 
{ 
    return uint64_t(v);
}

inline constexpr i32 operator "" _i32(unsigned long long v) 
{ 
    return int32_t(v);
}

inline constexpr i64 operator "" _i64(unsigned long long v) 
{ 
    return int64_t(v);
}

#ifndef MADRONA_GPU_MODE
using CountT = int64_t;
#else
using CountT = int32_t;
#endif

template<typename T>
concept EnumType = std::is_enum_v<T>;

}
