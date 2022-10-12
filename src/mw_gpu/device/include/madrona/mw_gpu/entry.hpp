/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <cstdint>

namespace madrona {
namespace mwGPU {

template <typename T>
__global__ void submitInit(uint32_t num_worlds, void *world_init_ptr)
{
    uint32_t invocation_idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (invocation_idx >= num_worlds) return;

    T::submitInit(invocation_idx, world_init_ptr);
}

template <typename T>
__global__ void submitRun(uint32_t num_worlds)
{
    uint32_t invocation_idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (invocation_idx >= num_worlds) return;

    T::submitRun(invocation_idx);
}

template <typename T,
          decltype(submitInit<T>) = submitInit<T>,
          decltype(submitRun<T>) = submitRun<T>>
struct EntryBase {};

}
}
