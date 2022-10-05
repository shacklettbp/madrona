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

static void launch()
{
    StateManager state_mgr;
    
    JobManager job_mgr(JobManager::makeEntry<Engine, CollisionSim>(
        [](Engine &ctx) {
            CollisionSim::entry(ctx);
        }), 0, 0, &state_mgr);
    
    job_mgr.waitForAllFinished();
}

}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    CollisionExample::launch();
}
