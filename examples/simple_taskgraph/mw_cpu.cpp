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

namespace SimpleTaskgraph {

static void launch(int num_worlds, uint64_t num_benchmark_ticks)
{
    // FIXME: Should have 1 initialization per world
    HeapArray<EnvInit> env_inits(num_worlds);
    for (int i = 0; i < num_worlds; i++) {
        env_inits.emplace(i, generateEnvironmentInitialization());
    }

    SimEntry entry(env_inits.data(), num_worlds);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < (int)num_benchmark_ticks; i++) {
        entry.run();
    }

    auto end = std::chrono::steady_clock::now();

    double duration = std::chrono::duration<double>(end - start).count();

    if (num_benchmark_ticks > 0) {
        printf("FPS: %f, Elapsed: %f\n", (double)num_benchmark_ticks *
            (double)num_worlds / duration, duration);
    }

    for (EnvInit &env_init : env_inits) {
        free(env_init.objsInit);
    }
}

}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s NUM_WORLDS NUM_TICKS", argv[0]);
        return EXIT_FAILURE;
    }

    int num_worlds = std::stoi(argv[1]);

    if (num_worlds < 1) {
        fprintf(stderr, "%s: num worlds must be > 0", argv[0]);
        return EXIT_FAILURE;
    }

    uint64_t num_benchmark_ticks = std::stoul(argv[2]);
    if (num_benchmark_ticks == 0) {
        fprintf(stderr, "%s: NUM_TICKS must be > 0", argv[0]);
    }

    SimpleTaskgraph::launch(num_worlds, num_benchmark_ticks);
}
