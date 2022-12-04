#include "simple.hpp"
#include "init.hpp"
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>

#include <cstdio>
#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace madrona;
using namespace std;

using namespace SimpleTaskgraph;

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s NUM_WORLDS NUM_TICKS\n", argv[0]);
        return EXIT_FAILURE;
    }

    int num_worlds = atoi(argv[1]);

    if (num_worlds < 1) {
        fprintf(stderr, "NUM_WORLDS must be >= 1");
        return EXIT_FAILURE;
    }

    uint64_t num_ticks = std::stoul(argv[2]);

    if (num_ticks == 0) {
        fprintf(stderr, "NUM_TICKS must be > 0");
        return EXIT_FAILURE;
    }

    // FIXME: this is really hacky
    HeapArray<EnvInit> env_inits(num_worlds);
    for (int i = 0; i < num_worlds; i++) {
        env_inits[i] = generateEnvironmentInitialization();

        ObjectInit *ptr;
        cudaMalloc(&ptr, env_inits[i].numObjs * sizeof(ObjectInit));
        cudaMemcpy(ptr, env_inits[i].objsInit,
                   env_inits[i].numObjs * sizeof(ObjectInit),
                   cudaMemcpyHostToDevice);

        free(env_inits[i].objsInit);
        env_inits[i].objsInit = ptr;
    }

    uint32_t render_width = 128;
    uint32_t render_height = 128;

    TrainingExecutor train_exec({
        .worldInitPtr = env_inits.data(),
        .numWorldInitBytes = sizeof(EnvInit),
        .numWorldDataBytes = sizeof(SimpleSim),
        .worldDataAlignment = alignof(SimpleSim),
        .numWorlds = uint32_t(num_worlds),
        .numExportedBuffers = 2,
        .gpuID = 0,
        .renderWidth = render_width,
        .renderHeight = render_height,
    }, {
        "SimpleExample::SimEntry",
        { SIMPLE_TASKGRAPH_SRC_LIST },
        { SIMPLE_TASKGRAPH_COMPILE_FLAGS },
        CompileConfig::OptMode::Debug,
        CompileConfig::Executor::TaskGraph,
    });

    void *agent_positions_gpu = train_exec.getExported(0);
    void *agent_rotations_gpu = train_exec.getExported(1);

    printf("%p %p\n", agent_positions_gpu, agent_rotations_gpu);

    uint64_t num_observation_bytes =
        num_worlds * render_width * render_height * 4;

    uint8_t *rgb_observations_gpu = train_exec.rgbObservations();
    uint8_t *rgb_observations_cpu =
        (uint8_t *)cu::allocReadback(num_observation_bytes);

    train_exec.run();

    cudaMemcpy(rgb_observations_cpu,
               rgb_observations_gpu,
               num_observation_bytes,
               cudaMemcpyDeviceToHost);

    stbi_write_bmp("/tmp/t.bmp", render_width, render_height,
                   4, rgb_observations_cpu);

    auto start = std::chrono::system_clock::now();
    for (int64_t i = 0; i < (int64_t)num_ticks; i++) {
        train_exec.run();
    }
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> diff = end - start;
    double elapsed = diff.count();

    printf("%f %f\n", double(num_ticks * num_worlds) / elapsed, elapsed);

    for (int i = 0; i < num_worlds; i++) {
        cudaFree(env_inits[i].objsInit);
    }
}
