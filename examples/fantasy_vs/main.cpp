/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/context.hpp>

#include <fstream>

#include "fvs.hpp"

using namespace madrona;

namespace fvs {

static void launch(const BenchmarkConfig &bench,
                   uint32_t num_benchmark_worlds)
{
    if (bench.enable) {
        DynArray<std::thread> threads(num_benchmark_worlds);
        HeapArray<std::chrono::time_point<std::chrono::steady_clock>> starts(
            num_benchmark_worlds);

        for (int i = 0; i < (int)num_benchmark_worlds; i++) {
            threads.emplace_back([&bench, &starts, i]() {
                StateManager state_mgr;

                JobManager job_mgr(JobManager::makeEntry<Engine>(
                    [&bench, &starts, i](Engine &ctx) {
                        starts[i] = std::chrono::steady_clock::now();
                        Game::entry(ctx, bench);
                    }), 1, 0, &state_mgr, false);

                job_mgr.waitForAllFinished();
            });
        }

        for (int i = 0; i < (int)num_benchmark_worlds; i++) {
            threads[i].join();
        }

        auto end = std::chrono::steady_clock::now();

        double duration = std::chrono::duration<double>(end - starts[0]).count();

        for (int i = 1; i < (int)num_benchmark_worlds; i++) {
            duration = std::max(std::chrono::duration<double>(end - starts[i]).count(), duration);
        }

        printf("Done\n");
        printf("FPS: %f, Elapsed: %f\n", (double)bench.numTicks * (double)num_benchmark_worlds / duration, duration);

    } else {
        StateManager state_mgr;

        JobManager job_mgr(JobManager::makeEntry<Engine>(
            [&bench](Engine &ctx) {
                Game::entry(ctx, bench);
            }), 0, 0, &state_mgr);

        job_mgr.waitForAllFinished();

        printf("Done\n");
    }
}

}

int main(int argc, char *argv[])
{
    bool benchmark_mode =
        argc > 1 && !strcmp(argv[1], "--bench");

    fvs::BenchmarkConfig bench { false, 0, 0, 0 };

    uint32_t num_benchmark_worlds = 0;
    if (benchmark_mode) {
        bench.enable = true;

        if (argc < 4) {
            FATAL("Usage: %s --bench NUM_TICKS NUM_WORLDS", argv[0]);
        }
        bench.numTicks = std::stoul(argv[2]);
        bench.numDragons = 50;
        bench.numKnights = 200;

        num_benchmark_worlds = std::stoul(argv[3]);
    }

    fvs::launch(bench, num_benchmark_worlds);
}
