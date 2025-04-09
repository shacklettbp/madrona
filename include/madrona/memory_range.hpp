/*
 * Copyright 2021-2023 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

namespace madrona {

// On the CPU backend, the MemoryRange is very trivial - just a simple
// pointer to malloc'd memory.
struct MemoryRange {
    enum Status : uint32_t {
        Allocated = 0,
        Freed = 255
    };

    static constexpr inline MemoryRange none()
    {
        return MemoryRange {
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
    CountT numElements;
    uint32_t gen;
    int32_t id;
#else
    CountT numElements;
    void *ptr;
#endif
};

inline bool operator==(MemoryRange a, MemoryRange b)
{
#ifdef MADRONA_GPU_MODE
    return a.numElements == b.numElements && a.gen == b.gen && a.id == b.id;
#else
    return a.ptr == b.ptr && a.numElements == b.numElements;
#endif
}

inline bool operator!=(MemoryRange a, MemoryRange b)
{
    return !(a == b);
}
    
}
