#include <madrona/math.hpp>

using namespace madrona;

struct alignas(16) PerspectiveCameraData {
    math::Vector3 position;
    math::Quat rotation;
    float xScale;
    float yScale;
    float zNear;
    int32_t worldIDX;
    uint32_t pad;
};

struct alignas(16) InstanceData {
    math::Vector3 position;
    math::Quat rotation;
    math::Diag3x3 scale;
    int32_t objectID;
    int32_t worldIDX;
};

struct BVHParams {
    // Given by the ECS
    InstanceData *instances;
    PerspectiveCameraData *views;
    int32_t *instanceOffsets;
    int32_t *viewOffsets;

    // Things which are required for the BVH build but the memory footprint
    // depends on quantities that only the ECS will know (i.e., num instances)
    uint32_t *mortonCodes;
};

namespace consts {
__constant__ BVHParams params;
}

// For now, just use #defines for parameterizing the kernels
#define BVH_MORTON_CODES_GRAIN_SIZE 8

// Because the number of instances / views isn't going to be known when the
// CPU launches this kernel (it just gets put into a CUgraph and run every
// frame), we are just going to max out the GPU and use a persistent thread
// approach to make sure all the work gets done.

// Stages of the top-level BVH build and ray cast
// 1) Generate morton codes
// 2) Sort the morton codes
// 3) Generate the internal nodes
// 4) Optimize the BVH
extern "C" __global__ void bvhEntry()
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Hello from BVH module!\n");
    }
}




