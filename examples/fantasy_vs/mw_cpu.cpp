#include <madrona/context.hpp>

#include <fstream>

#include "fvs.hpp"

using namespace madrona;

namespace fvs {

static void launch(int num_worlds, const BenchmarkConfig &bench)
{
    StateManager state_mgr(num_worlds);

    HeapArray<std::chrono::time_point<std::chrono::steady_clock>> starts(
        num_worlds);
    JobManager job_mgr(JobManager::makeEntry<Engine>(
        [&bench, &starts](Engine &ctx) {
            starts[ctx.worldID()] = std::chrono::steady_clock::now();
            Game::entry(ctx, bench);
        }), 0, 0, &state_mgr);

    job_mgr.waitForAllFinished();
    auto end = std::chrono::steady_clock::now();

    double duration = std::chrono::duration<double>(end - starts[0]).count();

    for (int i = 1; i < num_worlds; i++) {
        duration = std::max(std::chrono::duration<double>(end - starts[i]).count(), duration);
    }

    printf("Done\n");
    
    if (bench.enable) {
        printf("FPS: %f, Elapsed: %f\n", (double)bench.numTicks * (double)num_worlds / duration, duration);
    }
}

}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        FATAL("Usage: %s NUM_WORLDS", argv[0]);
    }

    int num_worlds = std::stoi(argv[1]);

    if (num_worlds < 1) {
        FATAL("%s: num worlds must be greater than 0", argv[0]);
    }

    bool benchmark_mode =
        argc > 2 && !strcmp(argv[2], "--bench");

    fvs::BenchmarkConfig bench { false, 0, 0, 0 };

    if (benchmark_mode) {
        bench.enable = true;

        if (argc < 4) {
            FATAL("Usage: %s NUM_WORLDS --bench NUM_TICKS", argv[0]);
        }
        bench.numTicks = std::stoul(argv[3]);
        bench.numDragons = 50;
        bench.numKnights = 200;
    }

    fvs::launch(num_worlds, bench);
}
