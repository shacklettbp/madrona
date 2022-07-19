#pragma once

#include <cstdint>

namespace madrona {
namespace gpuTrain {

template <typename T>
__global__ void submitInit(uint32_t num_worlds)
{
    uint32_t invocation_idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (invocation_idx >= num_worlds) return;

    T::submitInit(invocation_idx);
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
