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
template <typename RangeMapUnit>
struct RangeMap {
    CountT numUnits;

private:
    inline RangeMap(CountT num_units, RangeMapUnit *ptr)
        : numUnits(num_units), ptr_(ptr)
    {
    }

    RangeMapUnit *ptr_;

    friend class Context;
};
    
}
