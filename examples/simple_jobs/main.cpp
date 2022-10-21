/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/context.hpp>

#include "simple.hpp"
#include "init.hpp"

#include <chrono>

using namespace madrona;

namespace SimpleExample {

static void launch(uint32_t num_benchmark_ticks)
{
    StateManager state_mgr;

    EnvInit env_init = generateEnvironmentInitialization(num_benchmark_ticks);
    
    auto start = std::chrono::steady_clock::now();
    JobManager job_mgr(JobManager::makeEntry<Engine>(
        [&env_init](Engine &ctx) {
            SimpleSim::entry(ctx, env_init);
        }), 0, 0, &state_mgr);
    
    job_mgr.waitForAllFinished();
    auto end = std::chrono::steady_clock::now();

    double duration = std::chrono::duration<double>(end - start).count();

    if (num_benchmark_ticks > 0) {
        printf("FPS: %f, Elapsed: %f\n",
               (double)num_benchmark_ticks / duration, duration);
    }

    free(env_init.objsInit);
}

}

int main(int argc, char *argv[])
{
    uint64_t num_benchmark_ticks = 0;
    if (argc >= 2 && !strcmp(argv[1], "--bench")) {
        if (argc == 2) {
            fprintf(stderr, "Usage: %s [--bench NUM_TICKS]\n", argv[0]);
            return EXIT_FAILURE;
        }

        num_benchmark_ticks = std::stoul(argv[2]);
        if (num_benchmark_ticks == 0) {
            fprintf(stderr, "%s: NUM_TICKS must be > 0 in benchmark mode", argv[0]);
            return EXIT_FAILURE;
        }
    }

    SimpleExample::launch(num_benchmark_ticks);
}
