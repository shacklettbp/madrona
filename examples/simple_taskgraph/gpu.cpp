#include "simple.hpp"
#include "init.hpp"
#include <madrona/mw_gpu.hpp>

#include <cstdio>
#include <cuda_runtime.h>

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

    TrainingExecutor train_exec({
        .worldInitPtr = env_inits.data(),
        .numWorldInitBytes = sizeof(EnvInit),
        .numWorldDataBytes = sizeof(SimpleSim),
        .worldDataAlignment = alignof(SimpleSim),
        .numWorlds = uint32_t(num_worlds),
        .gpuID = 0,
    }, {
        "SimpleExample::SimEntry",
        { SIMPLE_TASKGRAPH_SRC_LIST },
        { SIMPLE_TASKGRAPH_COMPILE_FLAGS },
        CompileConfig::OptMode::Debug,
        CompileConfig::Executor::TaskGraph,
    });

    train_exec.run();

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
