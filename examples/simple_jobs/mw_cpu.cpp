/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/context.hpp>

#include <fstream>

#include "simple.hpp"
#include "init.hpp"

using namespace madrona;

namespace SimpleExample {

static void launch(int num_worlds)
{
    StateManager state_mgr(num_worlds);

    // FIXME: Should have 1 initialization per world
    EnvInit env_init = generateEnvironmentInitialization();

    JobManager job_mgr(JobManager::makeEntry<Engine>(
        [&env_init](Engine &ctx) {
            SimpleSim::entry(ctx, env_init);
        }), 0, 0, &state_mgr);

    job_mgr.waitForAllFinished();

    free(env_init.objsInit);
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

    SimpleExample::launch(num_worlds);
}
