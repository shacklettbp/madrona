#include "simple.hpp"
#include "init.hpp"
#include <madrona/mw_gpu.hpp>

#include <cstdio>
#include <cuda_runtime.h>

using namespace madrona;
using namespace std;

using namespace SimpleExample;

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s NUM_WORLDS\n", argv[0]);
        return EXIT_FAILURE;
    }

    int num_worlds = atoi(argv[1]);

    if (num_worlds < 1) {
        fprintf(stderr, "# Worlds must be >= 1");
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

    TrainingExecutor train_exec({
        .numWorldDataBytes = sizeof(SimpleSim),
        .worldDataAlignment = alignof(SimpleSim),
        .numWorlds = uint32_t(num_worlds),
        .gpuID = 0,
    }, {
        "SimpleExample::GPUEntry",
        { SIMPLE_EX_SRC_LIST },
        { SIMPLE_EX_COMPILE_FLAGS },
    });

    constexpr int64_t num_ticks = 1000; 

    auto start = std::chrono::system_clock::now();
    for (int64_t i = 0; i < num_ticks; i++) {
        train_exec.run();
    }
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> diff = end - start;
    double elapsed = diff.count();

    printf("%f %f\n", double(num_ticks * 1024) / elapsed, elapsed);

    for (int i = 0; i < num_worlds; i++) {
        cudaFree(env_inits[i].objsInit);
    }
}
