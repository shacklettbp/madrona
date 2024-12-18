/*
 * Copyright 2021-2023 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

namespace madrona {

// On the CPU backend, the RangeMap is very trivial - just a simple
// pointer to malloc'd memory.
struct RangeMap {
    enum Status : uint32_t {
        Allocated = 0,
        Freed = 1
    };

    static constexpr inline RangeMap none()
    {
        return RangeMap {
#ifdef MADRONA_GPU_MODE
            0,
            0xFFFF'FFFF_u32,
            0xFFFF'FFFF_i32,
#else
            0,
            nullptr,
#endif
        };
    }

#ifdef MADRONA_GPU_MODE
    CountT numUnits;
    uint32_t gen;
    int32_t id;
#else
    CountT numUnits;
    void *ptr;
#endif
};

inline bool operator==(RangeMap a, RangeMap b)
{
#ifdef MADRONA_GPU_MODE
    return a.numUnits == b.numUnits && a.gen == b.gen && a.id == b.id;
#else
    return a.ptr == b.ptr && a.numUnits == b.numUnits;
#endif
}

inline bool operator!=(RangeMap a, RangeMap b)
{
    return !(a == b);
}
    
}
