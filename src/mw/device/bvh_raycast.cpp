#define MADRONA_MWGPU_MAX_BLOCKS_PER_SM 4

#include <madrona/bvh.hpp>

using namespace madrona;

namespace sm {

// Only shared memory to be used
extern __shared__ uint8_t buffer[];

}

extern "C" __constant__ BVHParams bvhParams;

extern "C" __global__ void bvhRaycastEntry()
{
    printf("Hello from raycast! Camera data at %p\n",
            bvhParams.views);
}
