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
    void *taskGraph;
    void *stateManagerAddr;
    void *worldDataAddr;
    void *hostAllocatorAddr;
    void *hostPrintAddr;
    void *tmpAllocatorAddr;
    void *deviceTracingAddr;
    void *meshBVHsAddr;
    void *bvhInternalData;
    uint32_t numWorldDataBytes;
    uint32_t numWorlds;
    uint32_t jobGridsOffset;
    uint32_t jobListOffset;
    uint32_t maxJobsPerGrid;
    uint32_t sharedJobTrackerOffset;
    uint32_t userJobTrackerOffset;
    uint32_t numMeshBVHs;
    uint32_t raycastOutputWidth;
    uint32_t raycastOutputHeight;
    uint32_t raycastRGBD;

    static inline GPUImplConsts & get();
};

}
}

#ifdef MADRONA_GPU_MODE
extern "C" {
extern __constant__ madrona::mwGPU::GPUImplConsts
    madronaMWGPUConsts;
}

namespace madrona {
namespace mwGPU {

GPUImplConsts & GPUImplConsts::get()
{
    return madronaMWGPUConsts;
}

}
}

#endif
