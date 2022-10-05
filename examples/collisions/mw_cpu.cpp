/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/context.hpp>

#include <fstream>

#include "collisions.hpp"

using namespace madrona;

namespace CollisionExample {

static void launch(int num_worlds)
{
    StateManager state_mgr(num_worlds);

    JobManager job_mgr(JobManager::makeEntry<Engine, CollisionSim>(
        [](Engine &ctx) {
            CollisionSim::entry(ctx);
        }), 0, 0, &state_mgr);

    job_mgr.waitForAllFinished();
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

    CollisionExample::launch(num_worlds);
}
