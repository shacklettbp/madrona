#pragma once 

#include <cstdint>

namespace madrona {
namespace mwGPU {

struct GPUImplConsts {
    void *jobSystemAddr;
    void *stateManagerAddr;
    void *ctxDataAddr;
    uint32_t numCtxDataBytes;
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
