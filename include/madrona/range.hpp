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

#ifdef MADRONA_GPU_MODE
    CountT numUnits;
    uint32_t gen;
    int32_t id;
#else
private:

    inline RangeMap(CountT num_units, void *ptr)
        : num_units_(num_units), ptr_(ptr)
    {
    }

    void *ptr_;
    CountT num_units_;

    friend class Context;
#endif
};
    
}
