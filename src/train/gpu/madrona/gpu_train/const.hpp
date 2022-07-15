#pragma once 

#include <cstdint>

namespace madrona {
namespace gpuTrain {

struct GPUImplConsts {
    void *jobSystemAddr;
    void *stateManagerAddr;
    void *worldDataAddr;
    uint32_t jobGridsOffset;
    uint32_t jobListOffset;
    uint32_t maxJobsPerGrid;
    uint32_t jobTrackerOffset;

    static inline GPUImplConsts & get();
};

}
}

#ifdef MADRONA_TRAIN_MODE
extern "C" {
extern __constant__ madrona::gpuTrain::GPUImplConsts
    madronaTrainGPUImplConsts;
}

namespace madrona {
namespace gpuTrain {

GPUImplConsts & GPUImplConsts::get()
{
    return madronaTrainGPUImplConsts;
}

}
}

#endif
