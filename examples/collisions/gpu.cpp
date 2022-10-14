#include "collisions.hpp"
#include <madrona/mw_gpu.hpp>

#include <cstdio>

using namespace madrona;
using namespace std;

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

    TrainingExecutor train_exec({
        .numWorlds = uint32_t(num_worlds),
        .gpuID = 0,
    }, {
        "Collisions::TrainingEntry",
        { COLLISIONS_SRC_LIST },
        { COLLISIONS_COMPILE_FLAGS },
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
}
