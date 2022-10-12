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

struct GPUImplConsts {
    void *jobSystemAddr;
    void *stateManagerAddr;
    void *worldDataAddr;
    uint32_t numWorldDataBytes;
    uint32_t jobGridsOffset;
    uint32_t jobListOffset;
    uint32_t maxJobsPerGrid;
    uint32_t jobTrackerOffset;

    static inline GPUImplConsts & get();
};

}
}

#ifdef MADRONA_GPU_MODE
extern "C" {
extern __constant__ madrona::mwGPU::GPUImplConsts
    madronaTrainGPUImplConsts;
}

namespace madrona {
namespace mwGPU {

GPUImplConsts & GPUImplConsts::get()
{
    return madronaTrainGPUImplConsts;
}

}
}

#endif
