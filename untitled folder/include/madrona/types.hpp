#pragma once

#include <cstdint>

namespace madrona {

inline constexpr uint32_t operator "" _u32(unsigned long long v) 
{ 
    return uint32_t(v);
}

inline constexpr uint64_t operator "" _u64(unsigned long long v) 
{ 
    return uint64_t(v);
}

inline constexpr int32_t operator "" _i32(unsigned long long v) 
{ 
    return int32_t(v);
}

inline constexpr int64_t operator "" _i64(unsigned long long v) 
{ 
    return int64_t(v);
}

#ifndef MADRONA_GPU_MODE
using CountT = int64_t;
#else
using CountT = int32_t;
#endif

}
