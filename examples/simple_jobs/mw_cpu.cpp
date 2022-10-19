/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/context.hpp>

#include <chrono>

#include "simple.hpp"
#include "init.hpp"

using namespace madrona;

namespace SimpleExample {

static void launch(int num_worlds, uint64_t num_benchmark_ticks)
{
    StateManager state_mgr(num_worlds);

    // FIXME: Should have 1 initialization per world
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
        printf("FPS: %f, Elapsed: %f\n", (double)num_benchmark_ticks *
            (double)num_worlds / duration, duration);
    }

    free(env_init.objsInit);
}

}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s NUM_WORLDS", argv[0]);
        return EXIT_FAILURE;
    }

    int num_worlds = std::stoi(argv[1]);

    if (num_worlds < 1) {
        fprintf(stderr, "%s: num worlds must be > 0", argv[0]);
        return EXIT_FAILURE;
    }

    uint64_t num_benchmark_ticks = 0;
    if (argc >= 3 && !strcmp(argv[2], "--bench")) {
        if (argc == 3) {
            fprintf(stderr, "Usage: %s NUM_WORLDS [--bench NUM_TICKS]\n", argv[0]);
            return EXIT_FAILURE;
        }

        num_benchmark_ticks = std::stoul(argv[3]);
        if (num_benchmark_ticks == 0) {
            fprintf(stderr, "%s: NUM_TICKS must be > 0 in benchmark mode", argv[0]);
            return EXIT_FAILURE;
        }
    }

    SimpleExample::launch(num_worlds, num_benchmark_ticks);
}
